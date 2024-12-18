from Core.Common.Logger import logger
from Core.Retriever.BaseRetriever import BaseRetriever
import asyncio
import json
from Core.Common.Utils import truncate_list_by_token_size
from collections import Counter
from Core.Retriever.RetrieverFactory import register_retriever_method
from Core.Prompt import QueryPrompt


class SubgraphRetriever(BaseRetriever):
    def __init__(self, **kwargs):

        config = kwargs.pop("config")
        super().__init__(config)
        self.mode_list = ["concatenate_information_return_list", "induced_subgraph_return_networkx", "k_hop_return_set"]
        self.type = "subgraph"
        for key, value in kwargs.items():
            setattr(self, key, value)

    @register_retriever_method(type="subgraph", method_name="concatenate_information_return_list")
    async def _find_relevant_subgraph_by_concatenate_information(self, seed: str):
        try:
            if seed is None: return None
            assert self.config.use_subgraphs_vdb
            subgraph_datas = await self.subgraphs_vdb.retrieval_subgraphs(seed, top_k=self.config.top_k, need_score=False)
            if not len(subgraph_datas):
                return None

            return subgraph_datas
        except Exception as e:
            logger.exception(f"Failed to find relevant subgraph: {e}")

    @register_retriever_method(type="subgraph", method_name="k_hop_return_set")
    async def _find_subgraph_by_k_hop(self, seed: list[str], k: int):
        try:
            if seed is None: return None
            subgraph_datas = await self.graph.find_k_hop_neighbors_batch(start_nodes=seed, k=k) # set
            if not len(subgraph_datas):
                return None

            return subgraph_datas
        except Exception as e:
            logger.exception(f"Failed to find relevant subgraph: {e}")

    @register_retriever_method(type="subgraph", method_name="induced_subgraph_return_networkx")
    def _find_subgraph_by_networkx(self, seed: list[str]):
        try:
            if seed is None: return None
            subgraph_datas = self.graph.get_induced_subgraph(nodes=seed)

            return subgraph_datas
        except Exception as e:
            logger.exception(f"Failed to find relevant subgraph: {e}")