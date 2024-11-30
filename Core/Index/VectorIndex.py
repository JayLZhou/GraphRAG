"""
Here is the vector index for our GraphRAG
@Author: Yingli
@Reference: 1. https://github.com/geekan/MetaGPT/blob/main/metagpt/rag/engines/simple.py
            2. https://github.com/alibaba/app-controller/blob/main/Core/Index/BaseIndex.py
@Building index:
    1. load index from storage
    2. load index from documents
    3. load index from nodes
"""
from Core.Common.Logger import logger
import os
from typing import Any
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import (
    BaseNode,
    Document
)
from colbert.data import Queries
from Core.Index import get_index
from Core.Common.Utils import mdhash_id
from Core.Index.Schema import ColBertIndexConfig
class VectorIndex():
    """VectorIndex is designed to be simple and straightforward.

    It is a lightweight and easy-to-use vector database for ANN search.
    """

    def __init__(
        self, config
    ) -> None:
        self._index = None
        self.config = config
        self._index = get_index(self.config)


    async def upsert(self, data: list[Any]):
        if len(data) == 0:
            logger.warning("No data needs to insert into the vector database")
            return 
        if isinstance(data, dict):
            logger.info(f"Start inserting {len(data)} documents into the vector database")
            list_data = [
            {
                "__id__": k,
                **{k1: v1 for k1, v1 in v.items()},
            }
            for k, v in data.items()
            ]
            documents = [Document(text = t['content'], doc_id = t['__id__'], metadata = {"entity_name": t['entity_name']}, excluded_embed_metadata_keys=["entity_name"]) for t in list_data]
            await self._update_index_from_documents(documents)

        elif isinstance(data[0], Document):
            await self._update_index_from_documents(data)
        elif isinstance(data[0], BaseNode):
            await self._update_index_from_nodes(data)
        elif isinstance(data[0], str) and not isinstance(self.config, ColBertIndexConfig):
            documents = [Document(text = t, key = mdhash_id(t)) for t in data]
            await self._update_index_from_documents(documents)
        elif isinstance(data[0], str) and isinstance(self.config, ColBertIndexConfig):
            await self._update_index_from_lists(data)
        else:
            logger.warning("The type of data is not supported")
    
    async def upsert_with_embedding(self, text: str, embedding: List[float], metadata: dict):
        await self._update_index_from_documents([Document(text = text, embedding = embedding, metadata = metadata)])
    
    def exist_index(self):
        
        return os.path.exists(self.config.persist_path)

    async def retrieval(self, query, top_k = None):
        if not top_k:
            top_k = self._get_retrieve_top_k()
        if isinstance(self.config, ColBertIndexConfig):
            return await self._index.query(query_str = query, top_k = top_k)
        else:
            retriever: BaseRetriever = self._index.as_retriever(similarity_top_k = top_k)
            nodes = await retriever.aretrieve(query)
            return nodes

    async def retrieve_batch(self, queries, top_k = None):
        if not top_k:
            top_k = self._get_retrieve_top_k()
        if isinstance(self.config, ColBertIndexConfig):
            try:
                if not isinstance(queries, Queries):
                    queries = Queries(data = queries)
                return self._index.query_batch(queries = queries, top_k = top_k)
            except Exception as e:
                logger.exception(f"fail to search {queries} for {e}")
                return []
    def _get_retrieve_top_k(self):
        return self.retriever_configs.top_k

    def _storage_index(self):
        self._index.storage_context.persist(persist_dir=self.config.persist_path)



    async def _update_index_from_documents(self, docs: list[Any]):
        refreshed_docs = self._index.refresh_ref_docs(docs)

        # the number of docs that are refreshed. if True in refreshed_docs, it means the doc is refreshed.
        logger.info("refresh index size is {}".format(len([True for doc in refreshed_docs if doc])))

    async def _update_index_from_nodes(self, nodes: list[Any]):

        self._index._build_index_from_nodes(nodes)

        logger.info("insert node to index is {}".format(len(nodes)))


    async def _update_index_from_lists(self, docs_list: list[str]):
        # Only used for Colbert
        self._index._build_index_from_list(docs_list)
        logger.info("insert to index is {}".format(len(docs_list)))
