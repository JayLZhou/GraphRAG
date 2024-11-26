from dataclasses import dataclass, asdict

@dataclass
class Entity:

    def __init__(self, entity_name: str, source_id: str, entity_type: str = "", description: str = "" ):
        self.entity_name = entity_name
        self.entity_type = entity_type
        self.description = description
        self.source_id = source_id

    @property
    def as_dict(self):
        return asdict(self)
class Relationship:

    def __init__(self, src_id: str, tgt_id: str, source_id: str, **kwargs):
        self.src_id = src_id
        self.tgt_id = tgt_id
        self.source_id = source_id
        self.relation_name = kwargs.get('relation_name', "")
        self.weight = kwargs.get('weight', 0.0) # for GraphRAG and LightRAG
        self.description = kwargs.get('description', "") # for GraphRAG and LightRAG
        self.keywords = kwargs.get('keywords', "") # for LightRAG
        self.rank = kwargs.get('rank', 0) # for LightRAG