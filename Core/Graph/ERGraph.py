import re
import asyncio
import json
from collections import defaultdict, Counter
from typing import Union, Any
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
    processing_phrases
)
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
from metagpt.provider.llm_provider_registry import create_llm_instance
from pydantic import model_validator
from Core.Index import (
    get_rag_embedding
)    
from Core.Index.Schema import (
    FAISSIndexConfig
)
from Core.Index.VectorIndex import VectorIndex
class ERGraph(BaseGraph):
   
    text_chunks: JsonKVStorage = JsonKVStorage()
    er_graph: NetworkXStorage = NetworkXStorage()
    
    @model_validator(mode="after")
    def _init_vectordb(cls, data):
        index_config = FAISSIndexConfig(persist_path="./storage", embed_model = get_rag_embedding())
        cls.entity_vdb = VectorIndex(index_config)
        cls.relationship_vdb = VectorIndex(index_config)
        return data

    async def _construct_graph(self, chunks: dict[str, TextChunk]):
        try:
            filtered_keys = await self.text_chunks.filter_keys(list(chunks.keys()))
            inserting_chunks = {key: value for key, value in chunks.items() if key in filtered_keys}
            ordered_chunks = list(inserting_chunks.items())
            async def extract_openie_from_triples(chunk_key_dp: tuple[str, TextChunk]):
                chunk_key, chunk_dp = chunk_key_dp
                entities = await self.named_entity_recognition(chunk_dp)
                triples =  await self._openie_post_ner_extract(chunk_dp, entities)
                return await self._organize_records(entities, triples, chunk_key)
            #TODO: support persist for the extracted enetities and tirples
            results = await asyncio.gather(*[extract_openie_from_triples(c) for c in ordered_chunks])
       
            await self._update_graph(results)

            # Augment the graph by ann searching & Store the similarity objects for indexing. 

            await self._augment_graph()
          
        
            # # ---------- commit upsertings and indexing
            # await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
    
        finally:
            logger.info("Consturcting graph finisihed")

    

    async def _augment_graph(self):
        pass 
    async def named_entity_recognition(self, passage: str):
        ner_messages = EntityPrompt.NER.format(user_input=passage)

        response_content = await self.llm.aask(ner_messages)
        response_content = prase_json_from_response(response_content)
        
        if 'named_entities' not in response_content:
            response_content = []
        else:
            response_content = response_content['named_entities']



        return response_content

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



    async def _organize_records(self, entities, triples, chunk_key: str):
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)

        for entity in entities :
            entity_name =  processing_phrases(entity)
            maybe_nodes[entity_name].append({
                "source_id": chunk_key,
            })

        for triple in triples: 
            relationship = Relationship(src_id = processing_phrases(triple[0]), 
                                        tgt_id = processing_phrases(triple[2]), 
                                        weight=1.0, source_id=chunk_key, relation_name = processing_phrases(triple[1]))
            maybe_edges[(relationship.src_id, relationship.tgt_id)].append(relationship)

        return dict(maybe_nodes), dict(maybe_edges)

    async def _handle_single_entity_extraction(self, record_attributes: list[str], chunk_key: str) -> Union[Entity, None]:
        if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
            return None

        entity_name = clean_str(record_attributes[1].upper())
        if not entity_name.strip():
            return None

        return Entity(
            entity_name = entity_name,
            entity_type = clean_str(record_attributes[2].upper()),
            description = clean_str(record_attributes[3]),
            source_id = chunk_key
        )

    async def _handle_single_relationship_extraction(self, record_attributes: list[str], chunk_key: str) -> Union[Relationship, None]:
        if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
            return None

        return Relationship(
            src_id=clean_str(record_attributes[1].upper()),
            tgt_id=clean_str(record_attributes[2].upper()),
            weight=float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0,
            description=clean_str(record_attributes[3]),
            source_id=chunk_key,
            keywords = clean_str(record_attributes[4]) if self.config.use_keywords else None
        )

    async def _update_graph(self, results: list):
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)
        for m_nodes, m_edges in results:
            for k, v in m_nodes.items():
                maybe_nodes[k].extend(v)
            for k, v in m_edges.items():
                maybe_edges[tuple(sorted(k))].extend(v)

        entities = await asyncio.gather(*[self._merge_nodes_then_upsert(k, v) for k, v in maybe_nodes.items()])
        import pdb
        pdb.set_trace()
        if self.entity_vdb is not None:
            data_for_vdb = {
                mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"],
                    "entity_name": dp["entity_name"],
                }
                for dp in entities
            }
            await self.entity_vdb.upsert(data_for_vdb)
   
        relationships = await asyncio.gather(*[self._merge_edges_then_upsert(k[0], k[1], v) for k, v in maybe_edges.items()])
        #TODO: For relationships 
        if self.relationship_vdb is not None:
            data_for_vdb = {
                mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"],
                    "entity_name": dp["entity_name"],
                }
                for dp in relationships
            }
            await self.relationship_vdb.upsert(data_for_vdb)

    async def _merge_nodes_then_upsert(self, entity_name: str, nodes_data: list[Entity]):
        existing_node = await self.er_graph.get_node(entity_name)
        existing_data = [[]] if existing_node is None else [
            existing_node["source_id"],
        ]

        source_id = GRAPH_FIELD_SEP.join(
            set(dp["source_id"] for dp in nodes_data) | set(existing_data[0])
        )

        node_data = dict(source_id = source_id)
 
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
                    node_data=dict(source_id=source_id)
                )
        edge_data = dict(weight=weight, source_id=source_id, relation_name = relation_name)
        await self.er_graph.upsert_edge(src_id, tgt_id, edge_data = edge_data)
        return {**edge_data, "src_id": src_id, "tgt_id": tgt_id}
  
        

    def _build_context_for_entity_extraction(self, content: str) -> dict:
        return dict(
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter = DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
            entity_types=",".join(DEFAULT_ENTITY_TYPES),
            input_text=content
        )

    # 


    async def local_query_lightrag(
        self,
        query
    ) -> str:
        context = None
        # use_model_func = global_config["llm_model_func"]
        kw_prompt = QueryPrompt.KEYWORDS_EXTRACTION.format(query=query)
        result = await self.llm.aask(kw_prompt)

    
        keywords_data = prase_json_from_response(result)
        keywords = keywords_data.get("low_level_keywords", [])
        keywords = ", ".join(keywords)
   
        if keywords:
            context = await self._build_local_query_context_with_keywords(keywords)
        if self.query_config.only_need_context:
            return context
        if context is None:
            return QueryPrompt.FAIL_RESPONSE
        sys_prompt_temp = QueryPrompt.RAG_RESPONSE
        sys_prompt = sys_prompt_temp.format(
            context_data=context, response_type= self.query_config.response_type
        )
        response = await self.llm.aask(
            query,
            system_msgs=[sys_prompt]
        )
        if len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )

        return response
    
    async def global_query_lightrag(self, query):
        context = None
        kw_prompt = QueryPrompt.KEYWORDS_EXTRACTION.format(query=query)
        result = await self.llm.aask(kw_prompt)

        
        keywords_data = prase_json_from_response(result)
        keywords = keywords_data.get("high_level_keywords", [])
        keywords = ", ".join(keywords)
       
        if keywords:
            context = await self._build_global_query_context(keywords)

        if self.query_config.only_need_context:
            return context
        if context is None:
            return QueryPrompt.FAIL_RESPONSE

        sys_prompt_temp = QueryPrompt.RAG_RESPONSE
        sys_prompt = sys_prompt_temp.format(
            context_data=context, response_type=self.query_config.response_type
        )
        response = await self.llm.aask(
            query,
            system_msgs=[sys_prompt]
        )
        if len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )

        return response

    def _extract_node(self):
        pass

    def _extract_relationship(self):
        pass

    def _exist_graph(self):
        pass


    async def _build_local_query_context_with_keywords(self, query):
        results = await self.entity_vdb.retrieval(query, top_k=self.query_config.top_k)

        if not len(results):
            return None
        node_datas = await asyncio.gather(
            *[self.er_graph.get_node(r["entity_name"]) for r in results]
        )
        if not all([n is not None for n in node_datas]):
            logger.warning("Some nodes are missing, maybe the storage is damaged")
        node_degrees = await asyncio.gather(
            *[self.er_graph.node_degree(r["entity_name"]) for r in results]
        )
        node_datas = [
            {**n, "entity_name": k["entity_name"], "rank": d}
            for k, n, d in zip(results, node_datas, node_degrees)
            if n is not None
        ]  
        # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
        use_text_units = await self._find_most_related_text_unit_from_entities(node_datas)
        use_relations = await self._find_most_related_edges_from_entities(node_datas)
        logger.info(
            f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
        )
        entites_section_list = [["id", "entity", "type", "description", "rank"]]
        for i, n in enumerate(node_datas):
            entites_section_list.append(
                [
                    i,
                    n["entity_name"],
                    n.get("entity_type", "UNKNOWN"),
                    n.get("description", "UNKNOWN"),
                    n["rank"],
                ]
            )
        entities_context = list_to_quoted_csv_string(entites_section_list)

        relations_section_list = [
            ["id", "source", "target", "description", "keywords", "weight", "rank"]
        ]
        for i, e in enumerate(use_relations):
            relations_section_list.append(
                [
                    i,
                    e["src_tgt"][0],
                    e["src_tgt"][1],
                    e["description"],
                    e["keywords"],
                    e["weight"],
                    e["rank"],
                ]
            )
        relations_context = list_to_quoted_csv_string(relations_section_list)

        text_units_section_list = [["id", "content"]]
        for i, t in enumerate(use_text_units):
            text_units_section_list.append([i, t["content"]])
        text_units_context = list_to_quoted_csv_string(text_units_section_list)
        return f"""
            -----Entities-----
            ```csv
            {entities_context}
            ```
            -----Relationships-----
            ```csv
            {relations_context}
            ```
            -----Sources-----
            ```csv
            {text_units_context}
            ```
        """
    
    async def _build_global_query_context(self, keywords):
        results = await self.relationship_vdb.retrieval(keywords, top_k=self.query_config.top_k)

        if not len(results): 
            return None

        edge_datas = await asyncio.gather(
            *[self.er_graph.get_edge(r["src_id"], r["tgt_id"]) for r in results]
        )

        if not all([n is not None for n in edge_datas]):
            logger.warning("Some edges are missing, maybe the storage is damaged")
        edge_degree = await asyncio.gather(
            *[self.er_graph.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
        )
        edge_datas = [
            {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
            for k, v, d in zip(results, edge_datas, edge_degree)
            if v is not None
        ]
        edge_datas = sorted(
            edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
        )
        edge_datas = truncate_list_by_token_size(
            edge_datas,
            key=lambda x: x["description"],
            max_token_size = self.query_config.max_token_for_global_context,
        )

        use_entities = await self._find_most_related_entities_from_relationships(edge_datas)
        use_text_units = await self._find_related_text_unit_from_relationships(edge_datas)
        logger.info(
            f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
        )
        relations_section_list = [
            ["id", "source", "target", "description", "keywords", "weight", "rank"]
        ]
        for i, e in enumerate(edge_datas):
            relations_section_list.append(
                [
                    i,
                    e["src_id"],
                    e["tgt_id"],
                    e["description"],
                    e["keywords"],
                    e["weight"],
                    e["rank"],
                ]
            )
        relations_context = list_to_quoted_csv_string(relations_section_list)

        entites_section_list = [["id", "entity", "type", "description", "rank"]]
        for i, n in enumerate(use_entities):
            entites_section_list.append(
                [
                    i,
                    n["entity_name"],
                    n.get("entity_type", "UNKNOWN"),
                    n.get("description", "UNKNOWN"),
                    n["rank"],
                ]
            )
        entities_context = list_to_quoted_csv_string(entites_section_list)

        text_units_section_list = [["id", "content"]]
        for i, t in enumerate(use_text_units):
            text_units_section_list.append([i, t["content"]])
        text_units_context = list_to_quoted_csv_string(text_units_section_list)

        return f"""
    -----Entities-----
    ```csv
    {entities_context}
    ```
    -----Relationships-----
    ```csv
    {relations_context}
    ```
    -----Sources-----
    ```csv
    {text_units_context}
    ```
    """

    async def _find_most_related_entities_from_relationships(self, edge_datas: list[dict]):
        entity_names = set()
        for e in edge_datas:
            entity_names.add(e["src_id"])
            entity_names.add(e["tgt_id"])

        node_datas = await asyncio.gather(
            *[self.er_graph.get_node(entity_name) for entity_name in entity_names]
        )

        node_degrees = await asyncio.gather(
            *[self.er_graph.node_degree(entity_name) for entity_name in entity_names]
        )
        node_datas = [
            {**n, "entity_name": k, "rank": d}
            for k, n, d in zip(entity_names, node_datas, node_degrees)
        ]

        node_datas = truncate_list_by_token_size(
            node_datas,
            key=lambda x: x["description"],
            max_token_size = self.query_config.max_token_for_local_context,
        )

        return node_datas


    async def _find_related_text_unit_from_relationships(
        self,
        edge_datas: list[dict]
    ):
        text_units = [
            split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
            for dp in edge_datas
        ]

        all_text_units_lookup = {}

        for index, unit_list in enumerate(text_units):
            for c_id in unit_list:
                if c_id not in all_text_units_lookup:
                    all_text_units_lookup[c_id] = {
                        "data": await self.text_chunks.get_by_id(c_id),
                        "order": index,
                    }

        if any([v is None for v in all_text_units_lookup.values()]):
            logger.warning("Text chunks are missing, maybe the storage is damaged")
        all_text_units = [
            {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
        ]
        all_text_units = sorted(all_text_units, key=lambda x: x["order"])
        all_text_units = truncate_list_by_token_size(
            all_text_units,
            key=lambda x: x["data"]["content"],
            max_token_size = self.query_config.max_token_for_text_unit,
        )
        all_text_units = [t["data"] for t in all_text_units]

        return all_text_units
    

    async def hybrid_query(self, query):
        low_level_context = None
        high_level_context = None

        kw_prompt_temp = QueryPrompt.KEYWORDS_EXTRACTION
        kw_prompt = kw_prompt_temp.format(query=query)

        result = await self.llm.aask(kw_prompt)
        json_text = prase_json_from_response(result)
        try:
            keywords_data = json.loads(json_text)
            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
            hl_keywords = ", ".join(hl_keywords)
            ll_keywords = ", ".join(ll_keywords)
        except json.JSONDecodeError:
            try:
                result = (
                    result.replace(kw_prompt[:-1], "")
                    .replace("user", "")
                    .replace("model", "")
                    .strip()
                )
                result = "{" + result.split("{")[1].split("}")[0] + "}"
                keywords_data = json.loads(result)
                hl_keywords = keywords_data.get("high_level_keywords", [])
                ll_keywords = keywords_data.get("low_level_keywords", [])
                hl_keywords = ", ".join(hl_keywords)
                ll_keywords = ", ".join(ll_keywords)
            # Handle parsing error
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                return QueryPrompt.FAIL_RESPONSE

        if ll_keywords:
            low_level_context = await self._build_local_query_context(
                ll_keywords
            
            )

        if hl_keywords:
            high_level_context = await self._build_global_query_context(
                hl_keywords
             
            )

        context = self.combine_contexts(high_level_context, low_level_context)

        if self.query_config.only_need_context:
            return context
        if context is None:
            return QueryPrompt.FAIL_RESPONSE


        sys_prompt_temp = QueryPrompt.RAG_RESPONSE
        sys_prompt = sys_prompt_temp.format(
            context_data=context, response_type=self.query_config.response_type
        )
        response = await self.llm.aask(
            query,
            system_msgs=[sys_prompt]
        )
        if len(response) > len(sys_prompt):
            response = (
                response.replace(sys_prompt, "")
                .replace("user", "")
                .replace("model", "")
                .replace(query, "")
                .replace("<system>", "")
                .replace("</system>", "")
                .strip()
            )
        return response