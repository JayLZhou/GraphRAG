from Core.Utils.YamlModel import YamlModel
from dataclasses import field


class QueryConfig(YamlModel):
    # model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    query_type: str = "qa"
    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 20
    nei_k: int = 3
    num_doc: int = 5  # Default parameter for the HippoRAG
    # naive search
    naive_max_token_for_text_unit: int = 12000
    use_keywords: bool = False
    use_communiy_info: bool = False
    # local search
    
    enable_local: bool = False
    local_max_token_for_text_unit: int = 4000  # 12000 * 0.33

    local_max_token_for_community_report: int = 3200  # 12000 * 0.27
    local_community_single_one: bool = False
    community_information: bool = False  # Open for MS-GraphRAG based method
    # global search
    global_min_community_rating: float = 0
    global_max_consider_community: float = 512
    global_max_token_for_community_report: int = 16384
    max_token_for_global_context: int = 4000
    global_special_community_map_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    use_global_query: bool = False # For LightRAG and GraphRAG
    enable_hybrid_query: bool = False # For LightRAG 
    # For IR-COT
    max_ir_steps: int = 2

    # For Hipporag
    augmentation_ppr: bool = False
    entities_max_tokens: int = 2000
    relationships_max_tokens: int = 2000

    # For RAPTOR
    tree_search: bool = False

    # For TOG
    depth: int = 3
    width: int = 3

    # For G-Retriever (GR)
    max_txt_len: int = 512
    topk_e: int = 3
    cost_e: float = 0.5

    # For Medical GraphRAG
    topk_entity: int = 10
    k_hop: int = 2
