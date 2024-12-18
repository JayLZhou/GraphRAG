from Core.Common.Constants import Retriever
from Core.Retriever import *


class MixRetriever:
    def __init__(self, retriever_context):
        self.context = retriever_context
        self.retrievers = {}
        self.register_retrievers()

    def register_retrievers(self):
        self.retrievers[Retriever.ENTITY] = EntityRetriever(**self.context.as_dict)
        self.retrievers[Retriever.COMMUNITY] = CommunityRetriever(**self.context.as_dict)
        self.retrievers[Retriever.CHUNK] = ChunkRetriever(**self.context.as_dict)
        self.retrievers[Retriever.RELATION] = RelationshipRetriever(**self.context.as_dict)
        self.retrievers[Retriever.SUBGRAPH] = SubgraphRetriever(**self.context.as_dict)

    async def retrieve_relevant_content(self, type: Retriever, mode: str, **kwargs):
        return await self.retrievers[type].retrieve_relevant_content(mode=mode, **kwargs)

    @property
    def llm(self):
        return self.context.llm

    @property
    def config(self):
        return self.context.config
