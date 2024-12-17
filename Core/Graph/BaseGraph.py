import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
import igraph as ig
import numpy as np

from lazy_object_proxy.utils import await_
from scipy.sparse import csr_matrix

from Core.Common.Logger import logger
from typing import List
from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Memory import Memory
from Core.Prompt import GraphPrompt
from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.EntityRelation import Entity, Relationship
from Core.Common.Utils import (clean_str, build_data_for_merge, csr_from_indices, csr_from_indices_list)
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Utils.MergeER import MergeEntity, MergeRelationship


class BaseGraph(ABC):

    def __init__(self, config, llm, encoder):
        self.working_memory: Memory = Memory()  # Working memory
        self.config = config  # Build graph config
        self.llm = llm  # LLM instance
        self.ENCODER = encoder  # Encoder
        self._graph = None

    async def build_graph(self, chunks, force: bool = False):
        """
        Builds or loads a graph based on the input chunks.

        Args:
            chunks: The input data chunks used to build the graph.
            force: Whether to re-build the graph
        Returns:
            The graph if it already exists, otherwise builds and returns the graph.
        """
        # Try to load the graph
        logger.info("Starting build graph for the given documents")

        is_exist = await self._load_graph(force)
        if force or not is_exist:
            await self._clear()
            # Build the graph based on the input chunks
            await self._build_graph(chunks)
            # Persist the graph into file
            await self._persist_graph(force)
        logger.info("✅ Finished the graph building stage")

    async def _load_graph(self, force: bool = False):
        """
        Try to load the graph from the file
        """
        return await self._graph.load_graph(force)

    @property
    def namespace(self):
        return None

    # TODO: Try to rewrite here, not now
    @namespace.setter
    def namespace(self, namespace):
        self._graph.namespace = namespace

    @property
    def entity_metakey(self):
        # For almost of graph, entity_metakey is "entity_name"
        return "entity_name"

    async def _merge_nodes_then_upsert(self, entity_name: str, nodes_data: List[Entity]):
        existing_node = await self._graph.get_node(entity_name)

        existing_data = build_data_for_merge(existing_node) if existing_node else defaultdict(list)
        # Groups node properties by their keys for upsert operation.
        upsert_nodes_data = defaultdict(list)
        for node in nodes_data:
            for node_key, node_value in node.as_dict.items():
                upsert_nodes_data[node_key].append(node_value)

        merge_description = (MergeEntity.merge_descriptions(existing_data["description"],
                                                            upsert_nodes_data[
                                                                "description"]) if self.config.enable_entity_description else None)

        description = (
            await self._handle_entity_relation_summary(entity_name, merge_description)
            if merge_description
            else ""
        )
        source_id = (MergeEntity.merge_source_ids(existing_data["source_id"],
                                                  upsert_nodes_data["source_id"]))

        new_entity_type = (MergeEntity.merge_types(existing_data["entity_type"], upsert_nodes_data[
            "entity_type"]) if self.config.enable_entity_type else "")

        node_data = dict(source_id=source_id, entity_name=entity_name, entity_type=new_entity_type,
                         description=description)

        # Upsert the node with the merged data
        await self._graph.upsert_node(entity_name, node_data=node_data)

    async def _merge_edges_then_upsert(self, src_id: str, tgt_id: str, edges_data: List[Relationship]) -> None:
        # Check if the edge exists and fetch existing data
        existing_edge = await self._graph.get_edge(src_id, tgt_id) if await self._graph.has_edge(src_id,
                                                                                                 tgt_id) else None

        existing_edge_data = build_data_for_merge(existing_edge) if existing_edge else defaultdict(list)

        # Groups node properties by their keys for upsert operation.
        upsert_edge_data = defaultdict(list)
        for edge in edges_data:
            for edge_key, edge_value in edge.as_dict.items():
                upsert_edge_data[edge_key].append(edge_value)

        source_id = (MergeRelationship.merge_source_ids(existing_edge_data["source_id"],
                                                        upsert_edge_data["source_id"]))

        total_weight = (MergeRelationship.merge_weight(existing_edge_data["weight"],
                                                       upsert_edge_data["weight"]))
        merge_description = (MergeRelationship.merge_descriptions(existing_edge_data["description"],
                                                                  upsert_edge_data[
                                                                      "description"]) if self.config.enable_edge_description else "")

        description = (
            await self._handle_entity_relation_summary((src_id, tgt_id), merge_description)
            if self.config.enable_edge_description
            else ""
        )

        keywords = (MergeRelationship.merge_keywords(existing_edge_data["keywords"],
                                                     upsert_edge_data[
                                                         "keywords"]) if self.config.enable_edge_keywords else "")

        relation_name = (MergeRelationship.merge_relation_name(existing_edge_data["relation_name"],
                                                               upsert_edge_data[
                                                                   "relation_name"]) if self.config.enable_edge_name else "")
        # Ensure src_id and tgt_id nodes exist
        for node_id in (src_id, tgt_id):
            if not await self._graph.has_node(node_id):
                # Upsert node with source_id and entity_name
                await self._graph.upsert_node(
                    node_id,
                    node_data=dict(source_id=source_id, entity_name=node_id, entity_type="", description="")
                )

        # Create edge_data with merged data
        edge_data = dict(weight=total_weight, source_id=source_id,
                         relation_name=relation_name, keywords=keywords, description=description, src_id=src_id,
                         tgt_id=tgt_id)
        # Upsert the edge with the merged data
        await self._graph.upsert_edge(src_id, tgt_id, edge_data=edge_data)

    @abstractmethod
    def _extract_entity_relationship(self, chunk_key_pair: tuple[str, TextChunk]):
        """
        Abstract method to extract entities and the relationships between their in the graph.

        This method should be implemented by subclasses to define how node relationships are extracted.
        """
        pass

    @abstractmethod
    def _build_graph(self, chunks):
        """
        Abstract method to build the graph based on the input chunks.

        Args:
            chunks: The input data chunks used to build the graph.

        This method should be implemented by subclasses to define how the graph is built from the input chunks.
        """
        pass

    async def augment_graph_by_similarity_search(self, entity_vdb, duplicate=False):
        logger.info("Starting augment the existing graph with similariy edges")

        ranking = {node: await entity_vdb.retrieval_nodes(query=node, graph = self._graph, top_k=self.config.similarity_top_k, need_score = True) for node in
              await self._graph.nodes()}

        kb_similarity = defaultdict(list)
        for key, rank in ranking.items():
            max_score = 0
    
            for idx, (ns, score) in enumerate(zip(rank[0], rank[1])):
                if idx == 0:
                    max_score = score
                if not duplicate and idx == 0:
                    continue
                kb_similarity[key].append((ns['entity_name'], score / max_score))

        maybe_edges = defaultdict(list)
        # Refactored second part using dictionary iteration and enumerate
        for src_id, nns in kb_similarity.items():
    
            for idx, (nn, score) in enumerate(nns):
       
                if score < self.config.similarity_threshold or idx >= self.config.similarity_top_k:
                    break
                if nn == src_id:
                    continue
                tgt_id = nn

                # No need source_id for this type of edges
                relationship = Relationship(src_id=clean_str(src_id),
                                            tgt_id=clean_str(tgt_id),
                                            source_id="N/A",
                                            weight=self.config.similarity_max * score, relation_name="similarity")
                maybe_edges[(relationship.src_id, relationship.tgt_id)].append(relationship)

        # Merge the edges
        maybe_edges_aug = defaultdict(list)
        for k, v in maybe_edges.items():
            maybe_edges_aug[tuple(sorted(k))].extend(v)
        logger.info(f"Augmenting graph with {len(maybe_edges_aug)} edges")
     
        await asyncio.gather(*[self._merge_edges_then_upsert(k[0], k[1], v) for k, v in maybe_edges.items()])
        await self._persist_graph()
        logger.info("✅ Finished augment the existing graph with similariy edges")


    async def __graph__(self, elements: list):
        """
        Build the graph based on the input elements.
        """
        # Initialize dictionaries to hold aggregated node and edge information
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)

        # Iterate through each tuple of nodes and edges in the input elements
        for m_nodes, m_edges in elements:
            # Aggregate node information
            for k, v in m_nodes.items():
                maybe_nodes[k].extend(v)

            # Aggregate edge information
            for k, v in m_edges.items():
                maybe_edges[tuple(sorted(k))].extend(v)

        # Asynchronously merge and upsert nodes
        await asyncio.gather(*[self._merge_nodes_then_upsert(k, v) for k, v in maybe_nodes.items()])

        # Asynchronously merge and upsert edges
        await asyncio.gather(*[self._merge_edges_then_upsert(k[0], k[1], v) for k, v in maybe_edges.items()])

    async def _handle_entity_relation_summary(self, entity_or_relation_name: str, description: str) -> str:
        """
           Generate a summary for an entity or relationship.

           Args:
               entity_or_relation_name (str): The name of the entity or relationship.
               description (str): The detailed description of the entity or relationship.

           Returns:
               str: The generated summary.
        """

        # Encode the description into tokens
        tokens = self.ENCODER.encode(description)

        # Check if the token length is within the maximum allowed tokens for summarization
        if len(tokens) < self.config.summary_max_tokens:
            return description
        # Truncate the description to fit within the maximum token limit
        use_description = self.ENCODER.decode(tokens[:self.config.llm_model_max_token_size])

        # Construct the context base for the prompt
        context_base = dict(
            entity_name=entity_or_relation_name,
            description_list=use_description.split(GRAPH_FIELD_SEP)
        )
        use_prompt = GraphPrompt.SUMMARIZE_ENTITY_DESCRIPTIONS.format(**context_base)
        logger.debug(f"Trigger summary: {entity_or_relation_name}")

        # Asynchronously generate the summary using the language model
        return await self.llm.aask(use_prompt, max_tokens=self.config.summary_max_tokens)

    async def _persist_graph(self, force = False):
        await self._graph.persist(force)

    async def nodes_data(self):
        return await self._graph.get_nodes_data()

    async def edges_data(self, need_content=True):
        return await self._graph.get_edges_data(need_content)

    async def subgraphs_data(self):
        return await self._graph.get_subgraph_from_same_chunk()

    async def node_metadata(self):
        return await self._graph.get_node_metadata()

    async def edge_metadata(self):
        return await self._graph.get_edge_metadata()

    async def subgraph_metadata(self):
        return await self._graph.get_subgraph_metadata()

    async def stable_largest_cc(self):
        if isinstance(self._graph, NetworkXStorage):
            return await self._graph.get_stable_largest_cc()
        else:
            logger.exception("**Only NETWORKX is supported for finding the largest connected component.** ")
            return None

    async def cluster_data_to_subgraphs(self, cluster_data: dict):
        if isinstance(self._graph, NetworkXStorage):

            await self._graph.cluster_data_to_subgraphs(cluster_data)
        else:
            logger.exception("**Only NETWORKX is supported for constructing the cluster <-> node mapping.** ")
            return None

    async def community_schema(self):
        return await self._graph.get_community_schema()

    async def get_node(self, node_id):
        return await self._graph.get_node(node_id)

    async def get_node_by_index(self, index):
        return await self._graph.get_node_by_index(index)

    async def get_edge_by_index(self, index):
        return await self._graph.get_edge_by_index(index)

    async def get_node_by_indices(self, node_idxs):
        return await asyncio.gather(
            *[self.get_node_by_index(node_idx) for node_idx in node_idxs]
        )

    async def get_edge_by_indices(self, edge_idxs):
        return await asyncio.gather(
            *[self.get_edge_by_index(edge_idx) for edge_idx in edge_idxs]
        )

    async def get_edge(self, src, tgt):
        return await self._graph.get_edge(src, tgt)

    async def nodes(self):
        return await self._graph.nodes()

    async def edges(self):
        return await self._graph.edges()

    async def node_degree(self, node_id):
        return await self._graph.node_degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str):
        return await self._graph.edge_degree(src_id, tgt_id)

    async def get_node_edges(self, source_node_id: str):
        return await self._graph.get_node_edges(source_node_id)

    @property
    def node_num(self):
        return self._graph.get_node_num()

    @property
    def edge_num(self):
        return self._graph.get_edge_num()

    def get_induced_subgraph(self, nodes: list[str]):
        return self._graph.get_induced_subgraph(nodes)

    async def get_entities_to_relationships_map(self, is_directed=False):
        if self.node_num == 0:
            return csr_matrix((0, 0))

        node_neighbors = {node: list(await self._graph.neighbors(node)) for node in await self._graph.nodes()}

        # Construct the row and column indices for the CSR matrix
        data = []
        for node, neighbors in node_neighbors.items():
            for neighbor in neighbors:
                # Get the edge index (assuming edge indices are unique)
                edge_index = self._graph.get_edge_index(node, neighbor)
                if edge_index == -1: continue
                node_index = await self._graph.get_node_index(node)
                data.append([node_index, edge_index])
                if not is_directed:
                    neighbor_index = await self._graph.get_node_index(neighbor)
                    data.append([neighbor_index, edge_index])

        # Get the number of nodes and edges
        node_count = self.node_num
        edge_count = self.edge_num
        # Construct the CSR matrix
        return csr_from_indices(data, shape=(node_count, edge_count))

    async def get_relationships_attrs(self, key):
        if self.edge_num == 0:
            return []
        lists_of_attrs = []
        for edge in await self.edges_data(False):
            lists_of_attrs.append(edge[key])
        return lists_of_attrs

    async def get_relationships_to_chunks_map(self, doc_chunk):
        raw_relationships_to_chunks = await self.get_relationships_attrs(key="source_id")
        # Map Chunk IDs to indices

        raw_relationships_to_chunks = [
            [i for i in await doc_chunk.get_index_by_merge_key(chunk_ids) if i is not None]
            for chunk_ids in raw_relationships_to_chunks
        ]
        return csr_from_indices_list(
            raw_relationships_to_chunks, shape=(len(raw_relationships_to_chunks), await doc_chunk.size)
        )

    async def get_edge_weight(self, src_id: str, tgt_id: str):
        return await self._graph.get_edge_weight(src_id, tgt_id)

    async def get_node_index(self, node_key):
        return await self._graph.get_node_index(node_key)

    async def get_node_indices(self, node_keys):
        return await asyncio.gather(
            *[self.get_node_index(node_key) for node_key in node_keys]
        )

    async def personalized_pagerank(self, reset_prob_chunk, damping: float = 0.1):
        pageranked_probabilities = []
        igraph_ = ig.Graph.from_networkx(self._graph.graph)
        igraph_.es['weight'] = [await self.get_edge_weight(edge[0], edge[1]) for edge in list(await self.edges())]

        for reset_prob in reset_prob_chunk:
            pageranked_probs = igraph_.personalized_pagerank(vertices=range(self.node_num), damping=damping,
                                                             directed=False,
                                                             weights='weight', reset=reset_prob,
                                                             implementation='prpack')

            pageranked_probabilities.append(np.array(pageranked_probs))
        pageranked_probabilities = np.array(pageranked_probabilities)

        return pageranked_probabilities[0]

    async def get_neighbors(self, node_id: str):
        return await self._graph.neighbors(node_id)

    async def get_nodes(self):
        return await self._graph.nodes()

    async def find_k_hop_neighbors_batch(self, start_nodes: list[str], k: int):
        return await self._graph.find_k_hop_neighbors_batch(start_nodes=start_nodes, k=k)  # set

    async def get_edge_relation_name_batch(self, edges: list[tuple[str, str]]):
        return await self._graph.get_edge_relation_name_batch(edges=edges)

    async def _clear(self):
        self._graph.clear()

