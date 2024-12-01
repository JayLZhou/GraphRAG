"""
Index Config Factory.
"""
from Core.Index import get_rag_embedding
from Core.Index.Schema import (
    FAISSIndexConfig,
    ColBertIndexConfig,
    MilvusIndexConfig
)


class IndexConfigFactory:
    def __init__(self):
        self.creators = {
            "faiss": self._create_faiss_config,
            "milvus": self._create_milvus_config,
            "colbert": self._create_colbert_config,
        }

    def get_config(self, config, persist_path):
        """Key is PersistType."""
        return self.creators[config.vdb_type](config, persist_path)

    @staticmethod
    def _create_faiss_config(config, persist_path):
        return FAISSIndexConfig(
            persist_path=persist_path,
            embed_model=get_rag_embedding(config.embedding.api_type, config)
        )

    @staticmethod
    def _create_milvus_config(config, persist_path):
        return MilvusIndexConfig(persist_path=persist_path, embed_model=get_rag_embedding(config))

    @staticmethod
    def _create_colbert_config(config, persist_path):
        return ColBertIndexConfig(persist_path=persist_path, index_name="nbits_2",
                                  model_name=config.colbert_checkpoint_path, nbits=2)


get_index_config = IndexConfigFactory().get_config
