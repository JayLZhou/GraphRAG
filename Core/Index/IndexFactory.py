"""
RAG Index Factory.
@Reference: https://github.com/geekan/MetaGPT/blob/main/metagpt/rag/factories/index.py
@Provide: BM25, FaissVectorStore, and MilvusVectorStore
"""
import faiss
import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.base import BaseIndex
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.milvus import MilvusVectorStore
from Core.Index.ColBertStore import ColbertIndex
from pathlib import Path

from Core.Common.BaseFactory import ConfigBasedFactory
from Core.Index.Schema import (
    BaseIndexConfig,
    BM25IndexConfig,
    FAISSIndexConfig,
    MilvusIndexConfig,
    ColBertIndexConfig
)



class RAGIndexFactory(ConfigBasedFactory):
    def __init__(self):
        creators = {
            FAISSIndexConfig: self._create_faiss,
            BM25IndexConfig: self._create_bm25,
            MilvusIndexConfig: self._create_milvus,
            ColBertIndexConfig: self._crease_colbert
        }
        super().__init__(creators)

    def get_index(self, config: BaseIndexConfig, **kwargs) -> BaseIndex:
        """Key is PersistType."""
        return super().get_instance(config, **kwargs)

    def _create_faiss(self, config: FAISSIndexConfig, **kwargs) -> VectorStoreIndex:
        if os.path.exists(config.persist_path):
       
            vector_store = FaissVectorStore.from_persist_dir(str(config.persist_path))
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=config.persist_path)
            return self._index_from_storage(storage_context=storage_context, config=config, **kwargs)

        else:
            vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(config.embed_model.dimensions))
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            return  VectorStoreIndex(
            nodes = [],
            storage_context=storage_context,
            embed_model= config.embed_model,
        )


    def _create_bm25(self, config: BM25IndexConfig, **kwargs) -> VectorStoreIndex:
        storage_context = StorageContext.from_defaults(persist_dir=config.persist_path)

        return self._index_from_storage(storage_context=storage_context, config=config, **kwargs)

    def _create_milvus(self, config: MilvusIndexConfig, **kwargs) -> VectorStoreIndex:
        vector_store = MilvusVectorStore(collection_name=config.collection_name, uri=config.uri, token=config.token)

        return self._index_from_vector_store(vector_store=vector_store, config=config, **kwargs)


    def _crease_colbert(self, config: ColBertIndexConfig, **kwargs) -> VectorStoreIndex:
    #     import pdb
    #     # pdb.set_trace()
        index_path = (Path(config.persist_path) / config.index_name)
        if os.path.exists(index_path):
            return ColbertIndex.load_from_disk(config.persist_path, config.index_name)
        else:
           
            return ColbertIndex(**config.model_dump())
   
    def _index_from_storage(
        self, storage_context: StorageContext, config: BaseIndexConfig, **kwargs
    ) -> VectorStoreIndex:
        embed_model = self._extract_embed_model(config, **kwargs)

        return load_index_from_storage(storage_context=storage_context, embed_model=embed_model)

    def _index_from_vector_store(
        self, vector_store: BasePydanticVectorStore, config: BaseIndexConfig, **kwargs
    ) -> VectorStoreIndex:
        embed_model = self._extract_embed_model(config, **kwargs)

        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

    def _extract_embed_model(self, config, **kwargs) -> BaseEmbedding:
        return self._val_from_config_or_kwargs("embed_model", config, **kwargs)


get_index = RAGIndexFactory().get_index