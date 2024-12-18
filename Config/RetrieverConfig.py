from Core.Utils.YamlModel import YamlModel


class RetrieverConfig(YamlModel):
    # Retrieval Config
    query_type: str = "ppr"
    enable_local: bool = False
    use_entity_similarity_for_ppr: bool = True
    top_k_entity_for_ppr: int = 8
    node_specificity: bool = True
    damping: float = 0.1
    top_k: int = 5
    k_nei: int = 3
    node_specificity: bool = True
    damping: float = 0.1
    max_token_for_local_context: int = 4800  # 12000 * 0.4
    use_relations_vdb: bool = False
    use_subgraphs_vdb: bool = False