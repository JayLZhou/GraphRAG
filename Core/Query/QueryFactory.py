"""
Query Factory.
"""
from Core.Query.BaseQuery import BaseQuery
from Core.Query.BasicQuery import BasicQuery
from Core.Query.PPRQuery import PPRQuery
from Core.Query.KGPQuery import KGPQuery
from Core.Query.ToGQuery import ToGQuery
from Core.Query.GRQuery import GRQuery
from Core.Query.MedQuery import MedQuery

class QueryFactory:
    def __init__(self):
        self.creators = {
            "basic": self._create_base_query,
            "ppr": self._create_hippo_query,
            "kgp": self._create_kgp_query,
            "tog": self._create_tog_query,
            "gr": self._create_gr_query,
            "med":self._create_med_query,
        }

    def get_query(self, name, config, retriever) -> BaseQuery:
        """Key is PersistType."""
        return self.creators[name](config, retriever)

    @staticmethod
    def _create_base_query(config, retriever):
        return BasicQuery(config, retriever)

    @staticmethod
    def _create_hippo_query(config, retriever):
        return PPRQuery(config, retriever)

    @staticmethod
    def _create_kgp_query(config, retriever):
        return KGPQuery(config, retriever)

    @staticmethod
    def _create_tog_query(config, retriever):
        return ToGQuery(config, retriever)

    @staticmethod
    def _create_gr_query(config, retriever):
        return GRQuery(config, retriever)

    @staticmethod
    def _create_med_query(config, retriever):
        return MedQuery(config, retriever)

get_query = QueryFactory().get_query
