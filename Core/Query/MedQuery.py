from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Prompt import QueryPrompt
from typing import Union
import asyncio
# from torch_geometric.data import Data, InMemoryDataset
from typing import Any, Dict, List, Tuple, no_type_check
from pcst_fast import pcst_fast
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from Core.Common.Utils import truncate_str_by_token_size
from Core.Common.Constants import GRAPH_FIELD_SEP

class MedQuery(BaseQuery):
    def __init__(self, config, retriever_context):
        super().__init__(config, retriever_context)

    async def _concatenate_information(self, metagraph_relation: str, metagraph_edge: tuple[str, str]):
        metagraph_relation_seperated = metagraph_relation.split(GRAPH_FIELD_SEP)
        return list(map(lambda x: metagraph_edge[0] + " " + x + " " +metagraph_edge[1], metagraph_relation_seperated))


    async def _retrieve_relevant_contexts(self, query: str):
        # find the most relevant subgraph
        context = await self._retriever.retrieve_relevant_content(type=Retriever.SUBGRAPH, mode="concatenate_information_return_list", seed=query) # return list
        chunk_id = context[0]["source_id"]
        metagraph_nodes = set()
        iteration_count = 1
        while len(metagraph_nodes) < self.config.topk_entity:
            origin_nodes = await self._retriever.retrieve_relevant_content(seed=query,
                                                                           top_k=self.config.topk_entity * iteration_count,
                                                                           type=Retriever.ENTITY,
                                                                           mode="vdb")  # list[dict]
            for node in origin_nodes[(iteration_count - 1) * self.config.topk_entity:]:
                if node["source_id"] == chunk_id and node["entity_name"] not in metagraph_nodes:
                    metagraph_nodes.add(node["entity_name"])

            iteration_count += 1

        logger.info("Find top {} entities at iteration {}!".format(self.config.topk_entity, iteration_count))

        metagraph_nodes = await self._retriever.retrieve_relevant_content(seed=list(metagraph_nodes), k=self.config.k_hop, type=Retriever.SUBGRAPH, mode="k_hop_return_set") # return set
        metagraph = await self._retriever.retrieve_relevant_content(seed=list(metagraph_nodes), type=Retriever.SUBGRAPH, mode="induced_subgraph_return_networkx")  # return networkx
        metagraph_edges = list(metagraph.edges())
        metagraph_super_relations = await self._retriever.retrieve_relevant_content(seed=metagraph_edges, type=Retriever.RELATION, mode="by_source&target") # super relations
        zip_combined = tuple(zip(metagraph_super_relations, metagraph_edges))
        metagraph_concatenate = await asyncio.gather(*[self._concatenate_information(metagraph_super_relation, metagraph_edge) for metagraph_super_relation, metagraph_edge in zip_combined]) # list[list]

        context = ""
        for relations in metagraph_concatenate:
            context += ", ".join(relations)
            context += ", "
        return context



    async def query(self, query):
        context = await self._retrieve_relevant_contexts(query)
        print(context)
        response = await self.generation_qa(query, context)
        return response

    async def generation_qa(self, query: str, context: str):
        messages = [{"role": "system", "content": "You are an AI assistant that helps people find information."},
                    {"role": "user", "content": "the question is: " + query + ", the provided information is (list seperated by ,): " +  context}]
        response = await self.llm.aask(msg=messages)
        return response

    async def generation_summary(self, query, context):
        if context is None:
            return QueryPrompt.FAIL_RESPONSE