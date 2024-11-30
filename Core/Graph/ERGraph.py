import igraph as ig
import asyncio
import json
from collections import defaultdict, Counter
from typing import Union, Any
from scipy.sparse import csr_array
import numpy as np
from Core.Graph.BaseGraph import BaseGraph
from Core.Common.Logger import logger
from Core.Common.Utils import (
    clean_str,
    split_string_by_multi_markers,
    is_float_regex,
    list_to_quoted_csv_string,
    prase_json_from_response,
    truncate_list_by_token_size,
    mdhash_id,
    processing_phrases,
    min_max_normalize
)
from colbert.data import Queries

from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.Message import Message
from Core.Prompt import EntityPrompt, QueryPrompt
from Core.Schema.EntityRelation import Entity, Relationship
from Core.Storage.JsonKVStorage import JsonKVStorage
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Common.Constants import (
    GRAPH_FIELD_SEP, 
    DEFAULT_ENTITY_TYPES, 
    DEFAULT_RECORD_DELIMITER, 
    DEFAULT_COMPLETION_DELIMITER,
    DEFAULT_TUPLE_DELIMITER
)
from Core.Common.Memory import Memory
from Core.Community.ClusterFactory import get_community_instance
from Core.Common.QueryConfig import QueryConfig
from Core.Common.QueryConfig import query_config
from pydantic import model_validator
from Core.Index import (
    get_rag_embedding
)    
from Core.Index.Schema import (
     FAISSIndexConfig,
     ColBertIndexConfig
)
from tqdm import tqdm

from Core.Index.VectorIndex import VectorIndex
class ERGraph(BaseGraph):
   
    text_chunks: JsonKVStorage = JsonKVStorage()
    er_graph: NetworkXStorage = NetworkXStorage()
    chunk_key_to_idx: dict = defaultdict(int)
    @model_validator(mode="after")
    def _init_vectordb(cls, data):
        # index_config = FAISSIndexConfig(persist_path="./storage", embed_model = get_rag_embedding())
        index_config = ColBertIndexConfig(persist_path="./storage/colbert_index", index_name="nbits_2", model_name=cls.config.colbert_checkpoint_path, nbits=2)
        cls.entity_vdb = VectorIndex(index_config)
      

        return data

    async def _construct_graph(self, chunks: dict[str, TextChunk]):
        try:
            filtered_keys = await self.text_chunks.filter_keys(list(chunks.keys()))
            inserting_chunks = {key: value for key, value in chunks.items() if key in filtered_keys}
            self.ordered_chunks = list(inserting_chunks.items())       
            async def extract_openie_from_triples(chunk_idx: int, chunk_key_dp: tuple[str, TextChunk]):
                chunk_key, chunk_dp = chunk_key_dp
                entities = await self.named_entity_recognition(chunk_dp)
                triples =  await self._openie_post_ner_extract(chunk_dp, entities)
                # Maybe this is not the best way to organize the records
                self.chunk_key_to_idx[chunk_key] = chunk_idx
                return await self._organize_records(entities, triples, chunk_key)
            #TODO: support persist for the extracted enetities and tirples
            results = await asyncio.gather(*[extract_openie_from_triples(idx, chunk) for idx, chunk in enumerate(self.ordered_chunks)])
            print(self.chunk_key_to_idx)
            # Build graph based on the extracted entities and triples
            await self._add_to_graph(results)
            
            # Build the PPR context for the Hipporag algorithm
            await self._build_ppr_context()


            #Augment the graph by ann searching
            if self.config.enable_graph_augmentation:
                data_for_aug =  { mdhash_id(node, prefix="ent-"): node for node in self.er_graph.graph.nodes()}
                await self._augment_graph(queries = data_for_aug)

            # # ---------- commit upsertings and indexing
            # await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
    
        finally:
            logger.info("Consturcting graph finisihed")

    


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
        self.id_to_entity = defaultdict(int)

        nodes = list(self.er_graph.graph.nodes())
        edges = list(self.er_graph.graph.edges())

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
                edge_data = await self.er_graph.get_edge(edge[0], edge[1])
                source_ids = edge_data['source_id'].split(GRAPH_FIELD_SEP)
                for source_id in source_ids:
                    # Map document to edge
                    source_idx = self.chunk_key_to_idx[source_id]
                    edge_idx = edges.index(edge)
                    self.chunk_to_edge[(source_idx, edge_idx)] = 1

                # Map fact to phrases for both nodes in the edge
                node_idx_1 = nodes.index(edge[0])
                node_idx_2 = nodes.index(edge[1])
  
          

                self.edge_to_entity[(edge_idx, node_idx_1)] = 1
                self.edge_to_entity[(edge_idx, node_idx_2)] = 1

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


        self.chunk_to_edge_mat = csr_array(([int(v) for v in self.chunk_to_edge.values()], ([int(e[0]) for e in self.chunk_to_edge.keys()], [int(e[1]) for e in self.chunk_to_edge.keys()])),
                                      shape=(len(self.chunk_key_to_idx.keys()), len(edges)))

        self.edge_to_entity_mat = csr_array(([int(v) for v in  self.edge_to_entity.values()], ([e[0] for e in self.edge_to_entity.keys()], [e[1] for e in self.edge_to_entity.keys()])),
                                         shape=(len(edges), len(nodes)))
 
        self.chunk_to_entity_mat = self.chunk_to_edge_mat.dot(self.edge_to_entity_mat)
        self.chunk_to_entity_mat[self.chunk_to_entity_mat.nonzero()] = 1
        self.entity_doc_count = self.chunk_to_entity_mat.sum(0).T

    async def named_entity_recognition(self, passage: str):
        ner_messages = EntityPrompt.NER.format(user_input=passage)

        response_content = await self.llm.aask(ner_messages)
        entities = prase_json_from_response(response_content)
        
        if 'named_entities' not in entities:
            entities = []
        else:
            entities = entities['named_entities']
        return entities

    async def _openie_post_ner_extract(self, chunk, entities):
        named_entity_json = {"named_entities": entities}
        openie_messages = EntityPrompt.OPENIE_POST_NET.format(passage=chunk, named_entity_json=json.dumps(named_entity_json))
        response_content = await self.llm.aask(openie_messages)
        triples = prase_json_from_response(response_content)
        try:
            triples = triples["triples"]
        except:
           return []

        return triples



    async def _organize_records(self, entities, triples, chunk_key):
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)

        for entity in entities :
            entity_name =  processing_phrases(entity)
            maybe_nodes[entity_name].append({
                "source_id": chunk_key,
                "entity_name": entity_name
            })
      
 
            for triple in triples: 
                if len(triple) != 3:
                    logger.warning(f"triples length is not 3, triple is: {triple}, len is {len(triple)}, so skip it")
                    continue
                relationship = Relationship(src_id = processing_phrases(triple[0]), 
                                            tgt_id = processing_phrases(triple[2]), 
                                            weight=1.0, source_id = chunk_key, relation_name = processing_phrases(triple[1]))
                maybe_edges[(relationship.src_id, relationship.tgt_id)].append(relationship)

        return dict(maybe_nodes), dict(maybe_edges)

   

    async def _add_to_graph(self, results: list):
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)
        for m_nodes, m_edges in results:
            for k, v in m_nodes.items():
                maybe_nodes[k].extend(v)
            for k, v in m_edges.items():
                maybe_edges[tuple(sorted(k))].extend(v)
        
        entities = await asyncio.gather(*[self._merge_nodes_then_upsert(k, v) for k, v in maybe_nodes.items()])
 
        # If there is a vectordb, we need to upsert the entities
        if hasattr(self, "entity_vdb") and self.entity_vdb is not None:
            if not isinstance(self.entity_vdb.config, ColBertIndexConfig):
                data_for_vdb = {
                    mdhash_id(dp["entity_name"], prefix="ent-"): {
                        "content": dp["entity_name"],
                        "entity_name": dp["entity_name"],
                    }
                    for dp in entities
                }
            else:
                # If the colbert indexer is set, we need to upsert the entities
                data_for_vdb = [dp['entity_name'] for dp in entities]

            await self.entity_vdb.upsert(data_for_vdb)
            

        await asyncio.gather(*[self._merge_edges_then_upsert(k[0], k[1], v) for k, v in maybe_edges.items()])

    async def _augment_graph(self, queries, similarity_threshold = 0.8, similarity_top_k = 100, duplicate = True):
        """
        For each entity in the graph, get its synonyms from the knowledge base
        queries: list of enetity names
        """
        ranking = await self.entity_vdb.retrieve_batch(queries, top_k = similarity_top_k) 
        entity_names = list(queries.values())
        kb_similarity = {}
        for key, entity_name in queries.items():
            rank = ranking.data[key]
            filtered_rank = rank[1:] if duplicate else rank
            kb_similarity[entity_name] = (
                [entity_names[r[0]] for r in filtered_rank],
                [r[2] / rank[0][2] for r in filtered_rank]
            )

        maybe_edges = defaultdict(list)
        # Refactored second part using dictionary iteration and enumerate
        for src_id, nns in kb_similarity.items():
            processed_nns = [processing_phrases(nn) for nn in nns[0]]
            for idx, (nn, score) in enumerate(zip(processed_nns, nns[1])):
                if score < similarity_threshold or idx >= similarity_top_k:
                    break
                if nn == src_id:
                    continue
                tgt_id = nn

                # No need source_id for this type of edges
                relationship = Relationship(src_id = processing_phrases(src_id), 
                                        tgt_id = processing_phrases(tgt_id), 
                                        source_id = "N/A",
                                        weight= self.config.similarity_max * score, relation_name = "similarity")
                maybe_edges[(relationship.src_id, relationship.tgt_id)].append(relationship)

        # Merge the edges
        maybe_edges_aug = defaultdict(list)
        for k, v in maybe_edges.items():
            maybe_edges_aug[tuple(sorted(k))].extend(v)
        
        await asyncio.gather(*[self._merge_edges_then_upsert(k[0], k[1], v) for k, v in maybe_edges.items()])

              
    async def _merge_nodes_then_upsert(self, entity_name: str, nodes_data: list[Entity]):
        existing_node = await self.er_graph.get_node(entity_name)
        existing_data = [[]] if existing_node is None else [
            existing_node["source_id"],
        ]

        source_id = GRAPH_FIELD_SEP.join(
            set(dp["source_id"] for dp in nodes_data) | set(existing_data[0])
        )

        node_data = dict(source_id = source_id, entity_name = entity_name)
 
        await self.er_graph.upsert_node(entity_name, node_data=node_data)
        return {**node_data, "entity_name": entity_name}


    async def _merge_edges_then_upsert(self, src_id: str, tgt_id: str, edges_data: list[Relationship]):
        existing_edge_data = {}

        if await self.er_graph.has_edge(src_id, tgt_id):
            existing_edge_data = await self.er_graph.get_edge(src_id, tgt_id)

        #NOTE: For the nano-rag, it supports DSpy 
        weight = sum(dp.weight for dp in edges_data) + existing_edge_data.get("weight", 0)

        source_id = GRAPH_FIELD_SEP.join(
            set([dp.source_id for dp in edges_data] + split_string_by_multi_markers(existing_edge_data.get("source_id", ""), [GRAPH_FIELD_SEP])
        ))
        relation_name = GRAPH_FIELD_SEP.join(
            sorted(set(dp.relation_name for dp in edges_data) | {existing_edge_data.get("relation_name", "")})
        )
        for need_insert_id in (src_id, tgt_id):
            if not await self.er_graph.has_node(need_insert_id):
                await self.er_graph.upsert_node(
                    need_insert_id,
                    node_data=dict(source_id=source_id, entity_name = need_insert_id)
                )
        edge_data = dict(weight=weight, source_id=source_id, relation_name = relation_name)
        await self.er_graph.upsert_edge(src_id, tgt_id, edge_data = edge_data)
  
        

    def merge_elements_with_same_first_line(elements, prefix='Wikipedia Title: '):
        merged_dict = {}

        # Iterate through each element in the list
        for element in elements:
            # Split the element into lines and get the first line
            lines = element.split('\n')
            first_line = lines[0]

            # Check if the first line is already a key in the dictionary
            if first_line in merged_dict:
                # Append the current element to the existing value
                merged_dict[first_line] += "\n" + element.split(first_line, 1)[1].strip('\n')
            else:
                # Add the current element as a new entry in the dictionary
                merged_dict[first_line] = prefix + element

        # Extract the merged elements from the dictionary
        merged_elements = list(merged_dict.values())
        return merged_elements


  
    

    def reason_step(self, few_shot: list, query: str, passages: list, thoughts: list):
        """
        Given few-shot samples, query, previous retrieved passages, and previous thoughts, generate the next thought with OpenAI models. The generated thought is used for further retrieval step.
        :return: next thought
        """
        prompt_demo = ''
        for sample in few_shot:
            prompt_demo += f'{sample["document"]}\n\nQuestion: {sample["question"]}\nThought: {sample["answer"]}\n\n'

        prompt_user = ''

        #TODO: merge title for the hotpotQA dataset
        for passage in passages:
            prompt_user += f'{passage}\n\n'
        prompt_user += f'Question: {query} \n Thought:' + ' '.join(thoughts)


        try:
            response_content = self.llm.aask(msg = prompt_demo + prompt_user, system_msgs = QueryPrompt.IRCOT_REASON_INSTRUCTION)
        except Exception as e:
            print(e)
            return ''
        return response_content

    def get_colbert_max_score(self, query):
        queries_ = [query]
        encoded_query =self.entity_vdb._index.index_searcher.encode(queries_, full_length_search=False)
        encoded_doc =self.entity_vdb._index.index_searcher.checkpoint.docFromText(queries_).float()
        max_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()

        return max_score


    async def link_node_by_colbertv2(self, query_entities):
        entity_ids = []
        max_scores = []

        for query in query_entities:
            queries = Queries(path=None, data={0: query})

            queries_ = [query]
            # Only use for the colbert index
            if not isinstance(self.entity_vdb.config, ColBertIndexConfig):
                logger.error('The entity_vdb is not a ColBertIndexConfig')
            encoded_query = self.entity_vdb._index.index_searcher.encode(queries_, full_length_search=False)

            max_score = self.get_colbert_max_score(query)
     
            ranking = self.entity_vdb._index.index_searcher.search_all(queries, k=1)
  
            for entity_id, rank, score in ranking.data[0]:

                entity = self.id_to_entity[entity_id]
                entity_ = [entity]
                encoded_doc =self.entity_vdb._index.index_searcher.checkpoint.docFromText(entity_).float()
                real_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()

                entity_ids.append(entity_id)
                max_scores.append(real_score / max_score)

        # Create a vector (num_doc) with 1s at the indices of the retrieved documents and 0s elsewhere
        top_phrase_vec = np.zeros(len(self.er_graph.graph.nodes()))

        # Set the weight of the retrieved documents based on the number of documents they appear in
        for enetity_id in entity_ids:
            if self.config.node_specificity:
                if  self.entity_doc_count[enetity_id] == 0:
                    weight = 1
                else:
                    weight = 1 / self.entity_doc_count[enetity_id]
                top_phrase_vec[enetity_id] = weight
            else:
                top_phrase_vec[enetity_id] = 1.0
        return top_phrase_vec, {(query, self.id_to_entity[entity_id]): max_score for entity_id, max_score, query in zip(entity_ids, max_scores, query_entities)}
    async def rank_docs(self, query: str, top_k=10):
        """
        Rank documents based on the query using ColBERTv2 and PPR.
        @param query: the input phrase
        @param top_k: the number of documents to return
        @return: the ranked document ids and their scores
        """
        
        assert isinstance(query, str), 'Query must be a string'
        query_entities = await self._extract_eneity_from_query(query)

        # Use ColBERTv2 for retrieval with PPR score
        if len(query_entities) > 0:
            all_phrase_weights, linking_score_map = await self.link_node_by_colbertv2(query_entities)
            ppr_node_probs = await self._run_pagerank_igraph_chunk([all_phrase_weights])
            
        else:  # no entities found
            logger.warning('No entities found in query')
            ppr_chunk_prob = np.ones(len(self.extracted_triples)) / len(self.extracted_triples)

        # Combine scores using PPR
  
        edge_prob = self.edge_to_entity_mat.dot(ppr_node_probs)
        ppr_chunk_prob = self.chunk_to_edge_mat.dot(edge_prob)
        ppr_chunk_prob = min_max_normalize(ppr_chunk_prob)
 
        # Final document probability
        doc_prob = ppr_chunk_prob

        # Return top k documents
        sorted_doc_ids = np.argsort(doc_prob, kind='mergesort')[::-1]
        sorted_scores = doc_prob[sorted_doc_ids]
        
        
    
        return sorted_doc_ids.tolist()[:top_k], sorted_scores.tolist()[:top_k]
    
    async def _run_pagerank_igraph_chunk(self, reset_prob_chunk):
        """
        Run the PPR algorithm on a chunk of the graph.  
        @param reset_prob_chunk: a list of numpy arrays, each representing the PPR weights for a chunk of the graph    
        @return: a list of numpy arrays, each representing the PPR weights for the same chunk of the graph
        """
        pageranked_probabilities = []
        #TODO: as a method in our NetworkXGraph class or directly use the networkx graph
        # Transform the graph to igraph format 
        igraph_ = ig.Graph.from_networkx(self.er_graph.graph)
        igraph_.es['weight'] = [await self.er_graph.get_edge_weight(edge[0], edge[1]) for edge in list(self.er_graph.graph.edges())]

        for reset_prob in tqdm(reset_prob_chunk, desc='pagerank chunk'):
            pageranked_probs = igraph_.personalized_pagerank(vertices=range(len(self.er_graph.graph.nodes())), damping=self.config.damping, directed=False,
                                                            weights='weight', reset=reset_prob, implementation='prpack')

            pageranked_probabilities.append(np.array(pageranked_probs))
        pageranked_probabilities = np.array(pageranked_probabilities)
        
        return pageranked_probabilities[0]
    
    async def retrieve_step(self, query: str, top_k: int):
        ranks, scores = await self.rank_docs(query, top_k = top_k)
        # Extract passages from the corpus based on the ranked document ids
        retrieved_passages = [self.ordered_chunks[rank][1]["content"] for rank in ranks]

        return retrieved_passages, scores
    
    def _extract_node(self):
        pass

    def _extract_relationship(self):
        pass

    def _exist_graph(self):
        pass
    

    async def _extract_eneity_from_query(self, query):
        entities = []
        try:
            entities = await self.named_entity_recognition(query)

            entities = [processing_phrases(p) for p in entities]
        except:
            self.logger.error('Error in Query NER')

        return entities
    
    def _get_few_shot_examples(self) -> list:
        #TODO: implement the few shot examples
        return []
    
    async def query(self, query: str):
        
       # Initial retrieval step

        logger.info(f'Processing query: {query} at the first step')
        retrieved_passages, scores = await self.retrieve_step(query, query_config.top_k)
        thoughts = []
        passage_scores = {passage: score for passage, score in zip(retrieved_passages, scores)}
        few_shot_examples = self._get_few_shot_examples()
        # Iterative refinement loop
        for iteration in range(2, query_config.max_ir_steps + 1):
            logger.info("Entering the ir-cot iteration: {}".format(iteration))
            # Generate a new thought based on current passages and thoughts
            new_thought = await self.reason_step(few_shot_examples, query, retrieved_passages[ : query_config.top_k], thoughts)
            thoughts.append(new_thought)
            
            # Check if the thought contains the answer
            if 'So the answer is:' in new_thought:
                break
            

            # Retrieve new passages based on the new thought
            new_passages, new_scores = await self.retrieve_step(new_thought,  query_config.top_k)
            
            # Update passage scores
            for passage, score in zip(new_passages, new_scores):
                if passage in passage_scores:
                    passage_scores[passage] = max(passage_scores[passage], score)
                else:
                    passage_scores[passage] = score
            
            # Sort passages by score in descending order
            sorted_passages = sorted(
                passage_scores.items(), key=lambda item: item[1], reverse=True
            )
            retrieved_passages, scores = zip(*sorted_passages)