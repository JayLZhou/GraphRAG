import asyncio
from Core.Common.Utils import truncate_list_by_token_size
from Core.Retriever.BaseRetriever import BaseRetriever
from Core.Retriever.RetrieverFactory import register_retriever_method
from Core.Common.Logger import logger
import numpy as np


class RelationshipRetriever(BaseRetriever):
    def __init__(self, **kwargs):
        config = kwargs.pop("config")
        super().__init__(config)
        self.mode_list = ["entity_occurrence", "from_entity", "ppr", "vdb", "from_entity_by_agent", "get_all", "by_source&target"]
        self.type = "relationship"
        for key, value in kwargs.items():
            setattr(self, key, value)

    @register_retriever_method(type="relationship", method_name="ppr")
    async def _find_relevant_relationships_by_ppr(self, query, seed_entities: list[dict], node_ppr_matrix=None):
        #
        entity_to_edge_mat = await self._entities_to_relationships.get()
        if node_ppr_matrix is None:
            # Create a vector (num_doc) with 1s at the indices of the retrieved documents and 0s elsewhere
            node_ppr_matrix = await self._run_personalized_pagerank(query, seed_entities)
        edge_prob_matrix = entity_to_edge_mat.T.dot(node_ppr_matrix)
        topk_indices = np.argsort(edge_prob_matrix)[-self.config.top_k:]
        edges = await self.graph.get_edge_by_indices(topk_indices)

        return await self._construct_relationship_context(edges)

    @register_retriever_method(type="relationship", method_name="vdb")
    async def _find_relevant_relations_vdb(self, seed, need_score=False, need_context=True, top_k=None):
        try:
            if seed is None: return None
            assert self.config.use_relations_vdb
            if top_k is None:
                top_k = self.config.top_k

            edge_datas = await self.relations_vdb.retrieval_edges(query=seed, top_k=top_k, graph=self.graph,
                                                                  need_score=need_score)

            if not len(edge_datas):
                return None

            if need_context:
                edge_datas = await self._construct_relationship_context(edge_datas)
            return edge_datas
        except Exception as e:
            logger.exception(f"Failed to find relevant relationships: {e}")

    @register_retriever_method(type="relationship", method_name="from_entity")
    async def _find_relevant_relationships_from_entities(self, seed: list[dict]):
        all_related_edges = await asyncio.gather(
            *[self.graph.get_node_edges(node["entity_name"]) for node in seed]
        )
        all_edges = set()
        for this_edges in all_related_edges:
            all_edges.update([tuple(sorted(e)) for e in this_edges])
        all_edges = list(all_edges)
        all_edges_pack = await asyncio.gather(
            *[self.graph.get_edge(e[0], e[1]) for e in all_edges]
        )
        all_edges_degree = await asyncio.gather(
            *[self.graph.edge_degree(e[0], e[1]) for e in all_edges]
        )
        all_edges_data = [
            {"src_tgt": k, "rank": d, **v}
            for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
            if v is not None
        ]
        all_edges_data = sorted(
            all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
        )
        all_edges_data = truncate_list_by_token_size(
            all_edges_data,
            key=lambda x: x["description"],
            max_token_size=self.config.max_token_for_local_context,
        )
        return all_edges_data

    @register_retriever_method(type="relationship", method_name="from_entity_by_agent")
    async def _find_relevant_relations_by_entity_agent(self, query: str, entity: str, pre_relations_name=None,
                                                       pre_head=None, width=3):
        """
        Use agent to select the top-K relations based on the input query and entities
        Args:
            query: str, the query to be processed.
            entity: str, the entity seed
            pre_relations_name: list, the relation name that has already exists
            pre_head: bool, indicator that shows whether te pre head relations exist or not
            width: int, the search width of agent
        Returns:
            results: list[str], top-k relation candidates list
        """
        try:
            from Core.Common.Constants import GRAPH_FIELD_SEP
            from collections import defaultdict
            from Core.Prompt.TogPrompt import extract_relation_prompt

            # get relations from graph
            edges = await self.graph.get_node_edges(source_node_id=entity)
            relations_name_super_edge = await self.graph.get_edge_relation_name_batch(edges=edges)
            relations_name = list(map(lambda x: x.split(GRAPH_FIELD_SEP), relations_name_super_edge))  # [[], [], []]

            relations_dict = defaultdict(list)
            for index, edge in enumerate(edges):
                src, tar = edge[0], edge[1]
                for rel in relations_name[index]:
                    relations_dict[(src, rel)].append(tar)

            tail_relations = []
            head_relations = []
            for index, rels in enumerate(relations_name):
                if edges[index][0] == entity:
                    head_relations.extend(rels)  # head
                else:
                    tail_relations.extend(rels)  # tail

            if pre_relations_name:
                if pre_head:
                    tail_relations = list(set(tail_relations) - set(pre_relations_name))
                else:
                    head_relations = list(set(head_relations) - set(pre_relations_name))

            head_relations = list(set(head_relations))
            tail_relations = list(set(tail_relations))
            total_relations = head_relations + tail_relations
            total_relations.sort()  # make sure the order in prompt is always equal

            head_relations = set(head_relations)

            # agent
            prompt = extract_relation_prompt % (str(width), str(width),
                                                str(width)) + query + '\nTopic Entity: ' + entity + '\nRelations: There are %s relations provided in total, seperated by ;.' % str(
                len(
                    total_relations)) + '; '.join(
                total_relations) + ';' + "\nA: "

            result = await self.llm.aask(msg=[
                {"role": "system", "content": "You are an AI assistant that helps people find information."},
                {"role": "user", "content": prompt}
            ])

            # clean
            import re
            pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
            relations = []
            for match in re.finditer(pattern, result):
                relation = match.group("relation").strip()
                if ';' in relation:
                    continue
                score = match.group("score")
                if not relation or not score:
                    return False, "output uncompleted.."
                try:
                    score = float(score)
                except ValueError:
                    return False, "Invalid score"
                if relation in head_relations:
                    relations.append({"entity": entity, "relation": relation, "score": score, "head": True})
                else:
                    relations.append({"entity": entity, "relation": relation, "score": score, "head": False})

            if len(relations) == 0:
                flag = False
                logger.info("No relations found by entity: {} and query: {}".format(entity, query))
            else:
                flag = True

            # return
            if flag:
                return relations, relations_dict
            else:
                return [], relations_dict
        except Exception as e:
            logger.exception(f"Failed to find relevant relations by entity agent: {e}")

    @register_retriever_method(type="relationship", method_name="get_all")
    async def _get_all_relationships(self):
        edges = await self.graph.edges_data()
        return edges

    @register_retriever_method(type="relationship", method_name="by_source&target")
    async def _get_relationships_by_source_target(self, seed: list[tuple[str, str]]):
        return await self.graph.get_edge_relation_name_batch(edges=seed)