from pydantic import field_validator, ConfigDict
from Core.Common.LLMConfig import LLMConfig
from Core.Utils.YamlModel import YamlModel
from dataclasses import field


class QueryConfig(YamlModel):
    # model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    only_need_context: bool = False
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 20
    # naive search
    naive_max_token_for_text_unit: int = 12000
    # local search
    local_max_token_for_text_unit: int = 4000  # 12000 * 0.33
    max_token_for_local_context: int = 4800  # 12000 * 0.4
    local_max_token_for_community_report: int = 3200  # 12000 * 0.27
    local_community_single_one: bool = False
    # global search
    global_min_community_rating: float = 0
    global_max_consider_community: float = 512
    global_max_token_for_community_report: int = 16384
    max_token_for_global_context: int = 4000
    global_special_community_map_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )

    # For IR-COT
    max_ir_steps: int = 2
query_config = QueryConfig()