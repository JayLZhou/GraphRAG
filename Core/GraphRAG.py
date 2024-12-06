import asyncio
from collections import defaultdict
from typing import Union, Any

from lazy_object_proxy.utils import await_
from scipy.sparse import csr_matrix, csr_array
from pyfiglet import Figlet
from Core.Chunk.DocChunk import DocChunk
import numpy as np
from Core.Common.Constants import GRAPH_FIELD_SEP, ANSI_COLOR
from Core.Common.Logger import logger
import tiktoken
from Core.Schema.VdbResult import * 
from Core.Common.Utils import (mdhash_id, prase_json_from_response, clean_str, truncate_list_by_token_size, \
                               split_string_by_multi_markers)
from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict
from Core.Common.ContextMixin import ContextMixin
from Core.Graph.GraphFactory import get_graph
from Core.Index import get_index
from Core.Index.IndexConfigFactory import get_index_config
from Core.Prompt import GraphPrompt, QueryPrompt
from Core.Storage.NameSpace import Workspace
from Core.Community.ClusterFactory import get_community
from Core.Storage.PickleBlobStorage import PickleBlobStorage
from colorama import Fore, Style, init

init(autoreset=True)  # Initialize colorama and reset color after each print


class GraphRAG(ContextMixin, BaseModel):
    """A class representing a Graph-based Retrieval-Augmented Generation system."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    working_dir: str = Field(default="", exclude=True)

    # The following two matrices are utilized for mapping entities to their corresponding chunks through the specified link-path:
    # Entity Matrix: Represents the entities in the dataset.
    # Chunk Matrix: Represents the chunks associated with the entities.
    # These matrices facilitate the entity -> relationship -> chunk linkage, which is integral to the HippoRAG and FastGraphRAG models.

    @field_validator("working_dir", mode="before")
    @classmethod
    def check_working_dir(cls, value: str):
        if value == "":
            logger.error("Working directory cannot be empty")
        return value

    @model_validator(mode="before")
    def welcome_message(cls, values):

        f = Figlet(font='big')  #
        # Generate the large ASCII art text
        logo = f.renderText('DIGIMON')
        print(f"{Fore.GREEN}{'#' * 100}{Style.RESET_ALL}")
        # Print the logo with color
        print(f"{Fore.MAGENTA}{logo}{Style.RESET_ALL}")
        text = [
            "Welcome to DIGIMON: Deep Analysis of Graph-Based RAG Systems.",
            "",
            "Unlock advanced insights with our comprehensive tool for evaluating and optimizing RAG models.",
            "",
            "You can freely combine any graph-based RAG algorithms you desire. We hope this will be helpful to you!"
        ]

        # Function to print a boxed message
        def print_box(text_lines, border_color=Fore.BLUE, text_color=Fore.CYAN):
            max_length = max(len(line) for line in text_lines)
            border = f"{border_color}╔{'═' * (max_length + 2)}╗{Style.RESET_ALL}"
            print(border)
            for line in text_lines:
                print(
                    f"{border_color}║{Style.RESET_ALL} {text_color}{line.ljust(max_length)} {border_color}║{Style.RESET_ALL}")
            border = f"{border_color}╚{'═' * (max_length + 2)}╝{Style.RESET_ALL}"
            print(border)

        # Print the boxed welcome message
        print_box(text)

        # Add a decorative line for separation
        print(f"{Fore.GREEN}{'#' * 100}{Style.RESET_ALL}")
        return values

    @model_validator(mode="after")
    def _update_context(cls, data):
        cls.config = data.context.config
        cls.ENCODER = tiktoken.encoding_for_model(cls.config.token_model)
        cls.workspace = Workspace(data.working_dir, cls.config.exp_name)  # register workspace
        cls.graph = get_graph(data.config, llm=data.llm, encoder=cls.ENCODER)  # register graph
        cls.doc_chunk = DocChunk(data.config.chunk_method, cls.ENCODER, data.workspace.make_for("chunk_storage"))
        data = cls._init_storage_namespace(data)
        data = cls._register_vdbs(data)
        data = cls._register_community(data)
        data = cls._register_e2r_r2c_matrix(data)
        data = cls._register_retriever_context(data)
        return data

    @classmethod
    def _init_storage_namespace(cls, data):
        data.graph.namespace = data.workspace.make_for("graph_storage")
        if data.config.use_entities_vdb:
            data.entities_vdb_namespace = data.workspace.make_for("entities_vdb")
        if data.config.use_relations_vdb:
            data.relations_vdb_namespace = data.workspace.make_for("relations_vdb")
        if data.config.use_community:
            data.community_namespace = data.workspace.make_for("community_storage")
        if data.config.use_entity_link_chunk:
            data.e2r_namespace = data.workspace.make_for("map_e2r")
            data.r2c_namespace = data.workspace.make_for("map_r2c")

   
        return data

    @classmethod
    def _register_vdbs(cls, data):
        # If vector database is needed, register them into the class
        if data.config.use_entities_vdb:
            cls.entities_vdb = get_index(
                get_index_config(data.config, persist_path=data.entities_vdb_namespace.get_save_path()))
        if data.config.use_relations_vdb:
            cls.relations_vdb = get_index(
                get_index_config(data.config, persist_path=data.relations_vdb_namespace.get_save_path()))
        return data

    @classmethod
    def _register_community(cls, data):
        if data.config.use_community:
            cls.community = get_community(data.config.graph_cluster_algorithm,
                                          enforce_sub_communities=data.config.enforce_sub_communities, llm=data.llm,
                                         )
            cls.community.namesapce = data.community_namespace

        return data

    @classmethod
    def _register_e2r_r2c_matrix(cls, data):
        if data.config.use_entity_link_chunk:
            cls._entities_to_relationships = PickleBlobStorage(
                namespace=data.e2r_namespace, config=None
            )
            cls._relationships_to_chunks = PickleBlobStorage(
                namespace=data.r2c_namespace, config=None
            )
        return data

    @classmethod
    def _register_retriever_context(cls, data):
        # register the retriever context
        cls._retriever = None
        cls._retriever_context = defaultdict
        cls._retriever_context.update({"graph": True})
        cls._retriever_context.update({"chunks": True})
        cls._retriever_context.update({"entities_vdb": data.config.use_entities_vdb})
        cls._retriever_context.update({"relations_vdb": data.config.use_relations_vdb})
        cls._retriever_context.update({"community": data.config.use_community})
        cls._retriever_context.update({"community": data.config.use_entity_link_chunk})
        return data

    async def _extract_query_entities(self, query):
        entities = []
        try:
            ner_messages = GraphPrompt.NER.format(user_input=query)

            response_content = await self.llm.aask(ner_messages)
            entities = prase_json_from_response(response_content)

            if 'named_entities' not in entities:
                entities = []
            else:
                entities = entities['named_entities']

            entities = [clean_str(p) for p in entities]
        except Exception as e:
            logger.error('Error in Retrieval NER: {}'.format(e))

        return entities

    async def _extract_query_keywords(self, query):
        kw_prompt = QueryPrompt.KEYWORDS_EXTRACTION.format(query=query)
        result = await self.llm.aask(kw_prompt)

        keywords_data = prase_json_from_response(result)
        keywords = keywords_data.get("high_level_keywords", [])
        keywords = ", ".join(keywords)
        return keywords


    async def _build_retriever_context(self, query):
        """Build the retriever context for the given query."""
        logger.info("Building retriever context for the given query: {query}".format(query=query))

        self._retriever.register_context("graph", self.graph)
        self._retriever.register_context("query", query)
        for context_name, use_context in self._retriever_context.items():
            if use_context:
                self._retriever.register_context(context_name, getattr(self, context_name))

        if self.config.use_query_entity:
            try:
                query_entities = await self._extract_query_entities(query)
                self._retriever.register_context("query_entity", query_entities)
            except Exception as e:
                logger.error(f"Failed to extract query entities: {e}")
                raise

    async def _build_ppr_context(self):
        """
        Build the context for the Personalized PageRank (PPR) query.

        This function constructs two mappings:
            1. chunk_to_edge: Maps chunks (document sources) to edge indices.
            2. edge_to_entity: Maps edges to the entities (nodes) they connect.

        The function iterates over all edges in the graph, retrieves relevant metadata for each edge,
        and updates the mappings. These mappings are essential for executing PPR queries efficiently.
        """
        self.chunk_to_edge = defaultdict(int)
        self.edge_to_entity = defaultdict(int)
        self.entity_to_edge = defaultdict(int)
        self.id_to_entity = defaultdict(int)

        nodes = list(await self.graph.nodes())
        edges = list(await self.graph.edges())

        async def _build_edge_chunk_mapping(edge) -> None:
            """
            Build mappings for the edges of a given graph.

            Args:
                edge (Tuple[str, str]): A tuple representing the edge (node1, node2).
                edges (list): List of all edges in the graph.
                nodes (list): List of all nodes in the graph.
                docs_to_facts (Dict[Tuple[int, int], int]): Mapping of document indices to fact indices.
                facts_to_phrases (Dict[Tuple[int, int], int]): Mapping of fact indices to phrase indices.
            """
            try:
                # Fetch edge data asynchronously
                edge_data = await self.graph.get_edge(edge[0], edge[1])
                source_ids = edge_data['source_id'].split(GRAPH_FIELD_SEP)
                for source_id in source_ids:
                    # Map document to edge
                    source_idx = await self.doc_chunk.get_index_by_key(source_id)
                    edge_idx = edges.index((edge[0], edge[1]))
                    self.chunk_to_edge[(source_idx, edge_idx)] = 1

                # Map fact to phrases for both nodes in the edge
                node_idx_1 = nodes.index(edge[0])
                node_idx_2 = nodes.index(edge[1])

                self.edge_to_entity[(edge_idx, node_idx_1)] = 1
                self.entity_to_edge[(node_idx_1, edge_idx)] = 1
            except ValueError as ve:
                # Handle specific errors, such as when edge or node is not found
                logger.error(f"ValueError in edge {edge}: {ve}")
            except KeyError as ke:
                # Handle missing data in chunk_key_to_idx
                logger.error(f"KeyError in edge {edge}: {ke}")
            except Exception as e:
                # Handle general exceptions gracefully
                logger.error(f"Unexpected error processing edge {edge}: {e}")

        # Process all nodes asynchronously
        await asyncio.gather(*[_build_edge_chunk_mapping(edge) for edge in edges])

        for node in nodes:
            self.id_to_entity[nodes.index(node)] = node

        self.chunk_to_edge_mat = csr_array(([int(v) for v in self.chunk_to_edge.values()], (
            [int(e[0]) for e in self.chunk_to_edge.keys()], [int(e[1]) for e in self.chunk_to_edge.keys()])),
                                           shape=(await self.doc_chunk.size, len(edges)))

        self.edge_to_entity_mat = csr_array(([int(v) for v in self.edge_to_entity.values()], (
            [e[0] for e in self.edge_to_entity.keys()], [e[1] for e in self.edge_to_entity.keys()])),
                                            shape=(len(edges), len(nodes)))

        self.chunk_to_entity_mat = self.chunk_to_edge_mat.dot(self.edge_to_entity_mat)
      
        self.chunk_to_entity_mat[self.chunk_to_entity_mat.nonzero()] = 1
        self.entity_doc_count = self.chunk_to_entity_mat.sum(0).T

    async def build_e2r_r2c_maps(self, force = False):
        await self._build_ppr_context()
        logger.info("Starting build two maps: 1️⃣ entity <-> relationship; 2️⃣ relationship <-> chunks ")
        if not await self._entities_to_relationships.load(force):
            await self._entities_to_relationships.set(await self.graph.get_entities_to_relationships_map(False))
            await self._entities_to_relationships.persist()
        if not await self._relationships_to_chunks.load(force):
            await self._relationships_to_chunks.set(await self.graph.get_relationships_to_chunks_map(self.doc_chunk))
            await self._relationships_to_chunks.persist()
        logger.info("✅ Finished building the two maps ")
        
    async def insert(self, docs: Union[str, list[Any]]):

        """
        The main function that orchestrates the first step in the Graph RAG pipeline.
        This function is responsible for executing the various stages of the Graph RAG process,
        including chunking, graph construction, index building, and graph augmentation (optional).

        Configuration of the Graph RAG method is based on the parameters provided in the config file.
        For detailed information on the configuration and usage, please refer to the README.md.

        Args:
            docs (Union[str, list[[Any]]): A list of documents to be processed and inserted into the Graph RAG pipeline.
        """

        ####################################################################################################
        # 1. Chunking Stage
        ####################################################################################################
        logger.info("Starting chunk the given documents")
        await self.doc_chunk.build_chunks(docs)
        logger.info("✅ Finished the chunking stage")

        ####################################################################################################
        # 2. Building Graph Stage
        ####################################################################################################
        logger.info("Starting build graph for the given documents")
        await self.graph.build_graph(await self.doc_chunk.get_chunks())

        ####################################################################################################
        # 3. Index building Stage 
        # Data-driven content should be pre-built offline to ensure efficient online query performance.
        ####################################################################################################

        # NOTE: ** Ensure the graph is successfully loaded before proceeding to load the index from storage, as it represents a one-to-one mapping. **
        if self.config.use_entities_vdb:
            logger.info("Starting insert entities of the given graph into vector database")
            # await self.entities_vdb.build_index(await self.graph.nodes(), entity_metadata, force=False)
            await self.entities_vdb.build_index(await self.graph.nodes_data(), await self.graph.node_metadata(),
                                                force=False)
            logger.info("✅ Finished starting insert entities of the given graph into vector database")

        # Graph Augmentation Stage  (Optional) 
        # For HippoRAG and MedicalRAG, similarities between entities are utilized to create additional edges.
        # These edges represent similarity types and are leveraged in subsequent processes.

        # if self.config.enable_graph_augmentation:
        #     logger.info("Starting augment the existing graph with similariy edges")

        #     await self.graph.augment_graph_by_similrity_search()
        #     logger.info("✅ Finished augment the existing graph with similariy edges")

        if self.config.use_entity_link_chunk:
            await self.build_e2r_r2c_maps(True)

        if self.config.use_relations_vdb:
            logger.info("Starting insert relations of the given graph into vector database")
            relation_metadata = None
            for edge in await self.graph.edges():
                relation_metadata = {"src_id": edge["src_id"], "tgt_id": edge["tgt_id"]}
            await self.relations_vdb.build_index(await self.graph.edges_data(), relation_metadata, force=False)
            logger.info("✅ Finished starting insert relations of the given graph into vector database")

        if self.config.use_community:
            logger.info("Starting build community of the given graph")
            logger.start("Clustering nodes")
            await self.community.cluster(largest_cc=await self.graph.stable_largest_cc(),
                                         max_cluster_size=self.config.max_graph_cluster_size,
                                         random_seed=self.config.graph_cluster_seed)

            await self.community.generate_community_report(self.graph, False)
            logger.info("✅ [Community Report]  Finished")
         

         

    async def query(self, query):
        """
            Executes the query by extracting the relevant content, and then generating a response.
            Args:
                query: The query to be processed.
            Returns:
            """
        ####################################################################################################
        # 1. Building query relevant content (subgraph) Stage
        ####################################################################################################
        # await self._build_retriever_context(query)
        # await self._build_retriever_operator()
        query_entities = await self._extract_query_entities(query)
        entities = await self._link_entities(query_entities)
        # await self._link_chunks_from_entities(entities)
    
        # entities = await self._find_relevant_entities_vdb(query)
        # await self._find_relevant_chunks_from_entities(entities)
        keywords = await self._extract_query_keywords(query)
        relationships = await self._find_relevant_relations_vdb(query)
        print(relationships)

        ####################################################################################################
        # 2. Generation Stage
        ####################################################################################################

    # def get_colbert_max_score(self, query):

    async def _link_entities(self, query_entities):

        entities = []

        for query_entity in query_entities: 
            results = await self.entities_vdb.retrieval(query_entity, top_k=1)
            results = ColbertNodeResult(*results) if self.config.vdb_type == "colbert" else VectorIndexNodeResult(*results)
            node_datas = await results.get_node_data(self.graph) 
            # For entity link, we only consider the top-ranked entity
            entities.append(node_datas[0]) 
        return entities
      
      

    async def _find_relevant_entities_vdb(self, query, top_k=5):
        # ✅
        try:
            assert self.config.use_entities_vdb
            results = await self.entities_vdb.retrieval(query, top_k=top_k)             
            if not len(results):
                return None
            # We only support colbert and vector index (lamma-index based) for now
            results = ColbertNodeResult(*results) if self.config.vdb_type == "colbert" else VectorIndexNodeResult(*results)
            node_datas = await results.get_node_data(self.graph)
            if not all([n is not None for n in node_datas]):
                logger.warning("Some nodes are missing, maybe the storage is damaged")
            node_degrees = await asyncio.gather(
                *[self.graph.node_degree(node["entity_name"]) for node in node_datas]
            )
            node_datas = [
                {**n, "entity_name": n["entity_name"], "rank": d}
                for n, d in zip( node_datas, node_degrees)
                if n is not None
            ]
  
            return node_datas
        except Exception as e:
            logger.exception(f"Failed to find relevant entities_vdb: {e}")

    async def _find_relevant_relations_vdb(self, query, top_k=5):
        # ✅
        try:
            if query is None: return None
            assert self.config.use_relations_vdb
            results = await self.relations_vdb.retrieval(query, top_k=top_k)

            if not len(results):
                return None

            edge_datas = await asyncio.gather(
                *[self.graph.get_edge(r.metadata["src_id"], r.metadata["tgt_id"]) for r in results]
            )

            if not all([n is not None for n in edge_datas]):
                logger.warning("Some edges are missing, maybe the storage is damaged")
            edge_degree = await asyncio.gather(
                *[self.graph.edge_degree(r.metadata["src_id"], r.metadata["tgt_id"]) for r in results]
            )
            edge_datas = [
                {"src_id": k.metadata["src_id"], "tgt_id": k.metadata["tgt_id"], "rank": d, **v}
                for k, v, d in zip(results, edge_datas, edge_degree)
                if v is not None
            ]
            edge_datas = sorted(
                edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
            )
            edge_datas = truncate_list_by_token_size(
                edge_datas,
                key=lambda x: x["description"],
                max_token_size=self.config.max_token_for_global_context,
            )
            return edge_datas
        except Exception as e:
            logger.exception(f"Failed to find relevant relationships: {e}")

    async def _find_relevant_chunks_from_entities_relationships(self, node_datas: list[dict]):
        # ✅
        if len(node_datas) == 0:
            return None
        text_units = [
            split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
            for dp in node_datas
        ]
        edges = await asyncio.gather(
            *[self.graph.get_node_edges(dp["entity_name"]) for dp in node_datas]
        )
        all_one_hop_nodes = set()
        for this_edges in edges:
            if not this_edges:
                continue
            all_one_hop_nodes.update([e[1] for e in this_edges])
        all_one_hop_nodes = list(all_one_hop_nodes)
        all_one_hop_nodes_data = await asyncio.gather(
            *[self.graph.get_node(e) for e in all_one_hop_nodes]
        )
        all_one_hop_text_units_lookup = {
            k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
            for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
            if v is not None
        }
        all_text_units_lookup = {}
        for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
            for c_id in this_text_units:
                if c_id in all_text_units_lookup:
                    continue
                relation_counts = 0
                for e in this_edges:
                    if (
                            e[1] in all_one_hop_text_units_lookup
                            and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        relation_counts += 1
                all_text_units_lookup[c_id] = {
                    "data": await self.doc_chunk.get_data_by_key(c_id),
                    "order": index,
                    "relation_counts": relation_counts,
                }
        if any([v is None for v in all_text_units_lookup.values()]):
            logger.warning("Text chunks are missing, maybe the storage is damaged")
        all_text_units = [
            {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
        ]   
        # for node_data in node_datas:
        all_text_units = sorted(
            all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
        )
        all_text_units = truncate_list_by_token_size(
            all_text_units,
            key=lambda x: x["data"],
            max_token_size=self.config.local_max_token_for_text_unit,
        )
        all_text_units = [t["data"] for t in all_text_units]

        return all_text_units



    async def _find_relevant_relationships_from_entities(self, node_datas: list[dict]):
        # ✅
        all_related_edges = await asyncio.gather(
            *[self.graph.get_node_edges(node["entity_name"]) for node in node_datas]
        )
        all_edges = set()
        for this_edges in all_related_edges:
            all_edges.update([tuple(sorted(e)) for e in this_edges])
        all_edges = list(all_edges)
        all_edges_pack = await asyncio.gather(
            *[self.graph.get_edge(e[0], e[1]) for e in all_edges]
        )
        all_edges_degree = await asyncio.gather(
            *[self.graph.edge_degree(e[0], e[1]) for e in all_edges]
        )
        all_edges_data = [
            {"src_tgt": k, "rank": d, **v}
            for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
            if v is not None
        ]
        all_edges_data = sorted(
            all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
        )
        all_edges_data = truncate_list_by_token_size(
            all_edges_data,
            key=lambda x: x["description"],
            max_token_size=self.config.max_token_for_local_context,
        )
        return all_edges_data
