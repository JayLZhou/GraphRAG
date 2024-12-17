import html
import json
import os
from collections import defaultdict
from typing import Any, Union, cast
import networkx as nx
import numpy as np
from pydantic import model_validator
import asyncio
from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Logger import logger
from Core.Schema.CommunitySchema import LeidenInfo
from Core.Storage.BaseGraphStorage import BaseGraphStorage


class NetworkXStorage(BaseGraphStorage):
    def __init__(self):
        super().__init__()
        self.edge_list = None
        self.node_list = None

    name: str = "nx_data.graphml"  # The valid file name for NetworkX
    _graph: nx.Graph = nx.Graph()

    def load_nx_graph(self) -> bool:
        # Attempting to load the graph from the specified GraphML file
        logger.info(f"Attempting to load the graph from: {self.graphml_xml_file}")
        if os.path.exists(self.graphml_xml_file):
            try:
                self._graph = nx.read_graphml(self.graphml_xml_file)
                logger.info(
                    f"Successfully loaded graph from: {self.graphml_xml_file} with {self._graph.number_of_nodes()} nodes and {self._graph.number_of_edges()} edges")
                return True
            except Exception as e:
                logger.error(
                    f"Failed to load graph from: {self.graphml_xml_file} with {e}! Need to re-build the graph.")
                return False
        else:
            # GraphML file doesn't exist; need to construct the graph from scratch
            logger.info("GraphML file does not exist! Need to build the graph from scratch.")
            return False

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        logger.info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @model_validator(mode="after")
    def _register_node2emb(cls, data):
        cls._node_embed_algorithms = {
            "node2vec": data._node2vec_embed,
        }
        return data

    @property
    def graphml_xml_file(self):
        assert self.namespace is not None
        return self.namespace.get_save_path(self.name)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    async def load_graph(self, force: bool = False) -> bool:
        if force:
            logger.info("Force rebuilding the graph")
            return False
        else:
            return self.load_nx_graph()

    @property
    def graph(self):
        return self._graph

    async def _persist(self, force):
        if os.path.exists(self.graphml_xml_file) and not force:
            return
        logger.info(f"Writing graph into {self.graphml_xml_file}")
        NetworkXStorage.write_nx_graph(self.graph, self.graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        # [numberchiffre]: node_id not part of graph returns `DegreeView({})` instead of 0
        return self._graph.degree(node_id) if self._graph.has_node(node_id) else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return (self._graph.degree(src_id) if self._graph.has_node(src_id) else 0) + (
            self._graph.degree(tgt_id) if self._graph.has_node(tgt_id) else 0
        )

    async def get_edge_weight(
            self, source_node_id: str, target_node_id: str
    ) -> Union[float, None]:
        edge_data = self._graph.edges.get((source_node_id, target_node_id))
        return edge_data.get("weight") if edge_data is not None else None

    async def get_edge(
            self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    async def upsert_node(self, node_id: str, node_data: dict):
        self._graph.add_node(node_id, **node_data)

    # TODO: not use dict for edge_data
    async def upsert_edge(
            self, source_node_id: str, target_node_id: str, edge_data: dict
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):

        for node_id, clusters in cluster_data.items():
            self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)
        logger.info(f"Rewrite the graph with cluster data")
        await self._persist(force=True)

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        if algorithm not in self._node_embed_algorithms:
            raise ValueError(f"Node embedding algorithm {algorithm} not supported")
        return await self._node_embed_algorithms[algorithm]()

    async def _node2vec_embed(self):
        pass

    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a stable way.
        """
        from graspologic.utils import largest_connected_component

        graph = graph.copy()
        graph = cast(nx.Graph, largest_connected_component(graph))
        node_mapping = {node: html.unescape(node.upper().strip()) for node in graph.nodes()}  # type: ignore
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    async def persist(self, force):
        return await self._persist(force)

    async def get_nodes(self):
        return self._graph.nodes()


    # TODO: remove to the basegraph class
    async def get_nodes_data(self):
        node_list = list(self._graph.nodes())

        async def get_node_data(node_id):
            node_data = await self.get_node(node_id)
            node_data.setdefault("description", "")
            node_data.setdefault("entity_type", "")
            content_parts = []
            content_parts.append(node_data["entity_name"])

            if node_data["entity_type"]:
                content_parts.append(f"{node_data['entity_type']}")

            if node_data["description"]:
                content_parts.append(f"{node_data['description']}")

            node_data["content"] = ": ".join(content_parts) if content_parts else ""

            return node_data

        nodes = await asyncio.gather(*[get_node_data(node) for node in node_list])

        return nodes

    async def get_edges_data(self, need_content=True):
        edge_list = list(self._graph.edges())
        edges = []

        async def get_edge_data(edge_id):
            edge_data = await self.get_edge(edge_id[0], edge_id[1])
            if need_content:
                description = edge_data.get("description", "")
                relation_name = edge_data.get("relation_name", "")
                keywords = edge_data.get("keywords", "")
                if relation_name != "":
                    edge_data["content"] = relation_name
                else:
                    edge_data["content"] = "{keywords} {src_id} {tgt_id} {description}".format(
                        keywords=keywords, src_id=edge_data["src_id"], tgt_id=edge_data["tgt_id"],
                        description=description)
            edges.append(edge_data)

        await asyncio.gather(*[get_edge_data(edge) for edge in edge_list])
        return edges

    async def get_subgraph_from_same_chunk(self):
        # origin_nodes = await self.get_nodes_data() # list[dict]
        from Core.Common.Constants import GRAPH_FIELD_SEP
        origin_edges = await self.get_edges_data() # list[dict]

        from collections import defaultdict
        # chunk_to_metagraph_nodes = defaultdict(list)  # {"chunk_id": [node1, node2,...]}
        chunk_to_metagraph_edges = defaultdict(list)  # {"chunk_id": [edge1, edge2,...]}
        # for node in origin_nodes:
        #     chunk_to_metagraph_nodes[node['source_id']].append(node)
        for edge in origin_edges:
            chunk_to_metagraph_edges[edge["source_id"]].append(edge)

        subgraphs = []
        async def get_subgraph_data(key, value):
            subgraph_context = ""
            for ed in value:
                seperated_edge = ed["relation_name"].split(GRAPH_FIELD_SEP)
                tmp = tuple(map(lambda x: ed['src_id'] + " " + x + " " + ed["tgt_id"], seperated_edge))
                tmp = "; ".join(tmp)
                subgraph_context += tmp
                subgraph_context += "; "
            subgraphs.append({"source_id":key, "content": subgraph_context})

        await asyncio.gather(*[get_subgraph_data(key, value) for key, value in chunk_to_metagraph_edges.items()])

        return subgraphs







    async def get_stable_largest_cc(self):
        return NetworkXStorage.stable_largest_connected_component(self._graph)

    async def cluster_data_to_subgraphs(self, cluster_data):
        await self._cluster_data_to_subgraphs(cluster_data)

    async def get_community_schema(self):
        max_num_ids = 0
        levels = defaultdict(set)
        _schemas: dict[str, LeidenInfo] = defaultdict(LeidenInfo)
        for node_id, node_data in self._graph.nodes(data=True):
            if "clusters" not in node_data:
                continue
            clusters = json.loads(node_data["clusters"])
            this_node_edges = self._graph.edges(node_id)

            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                _schemas[cluster_key].level = level
                _schemas[cluster_key].title = f"Cluster {cluster_key}"
                _schemas[cluster_key].nodes.add(node_id)
                _schemas[cluster_key].edges.update(
                    [tuple(sorted(e)) for e in this_node_edges]
                )
                _schemas[cluster_key].chunk_ids.update(
                    node_data["source_id"].split(GRAPH_FIELD_SEP)
                )
                max_num_ids = max(max_num_ids, len(_schemas[cluster_key].chunk_ids))

        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            # compute the sub-communities by nodes intersection
            for comm in this_level_comms:
                _schemas[comm].sub_communities = [
                    c
                    for c in next_level_comms
                    if _schemas[c].nodes.issubset(_schemas[comm].nodes)
                ]

        for _, v in _schemas.items():
            v.edges = list(v.edges)
            v.edges = [list(e) for e in v.edges]
            v.nodes = list(v.nodes)
            v.chunk_ids = list(v.chunk_ids)
            v.occurrence = len(v.chunk_ids) / max_num_ids
        return _schemas

    async def get_node_metadata(self) -> list[str]:
        return ["entity_name"]

    async def get_edge_metadata(self) -> list[str]:
        relation_metadata = ["src_id", "tgt_id"]
        return relation_metadata

    async def get_subgraph_metadata(self) -> list[str]:
        return ["source_id"]

    def get_node_num(self):
        return self._graph.number_of_nodes()

    def get_edge_num(self):
        return self._graph.number_of_edges()

    async def nodes(self):
        return self._graph.nodes()

    async def edges(self):
        return self._graph.edges()

    async def neighbors(self, node_id):
        return self._graph.neighbors(node_id)

    def get_edge_index(self, src_id, tgt_id):
        if self.edge_list is None:
            self.edge_list = list(self._graph.edges())
        try:
            return self.edge_list.index((src_id, tgt_id))
        except ValueError:
            return -1

    async def get_induced_subgraph(self, nodes: list[str]):
        return self._graph.subgraph(nodes)

    async def get_node_index(self, node_id):
        if self.node_list is None:
            self.node_list = list(self._graph.nodes())
        try:
            return self.node_list.index(node_id)
        except ValueError:
            logger.error(f"Node {node_id} not in graph")

    async def get_node_by_index(self, index):
        if self.node_list is None:
            self.node_list = list(self._graph.nodes())
        return await self.get_node(self.node_list[index])

    async def get_edge_by_index(self, index):
        if self.edge_list is None:
            self.edge_list = list(self._graph.edges())

        return await self.get_edge(self.edge_list[index][0], self.edge_list[index][1])

    async def find_k_hop_neighbors(self, start_node: str, k: int) -> set:
        """
        find the k hop neighbors about the given input node
        :param start_node: str, entity_name
        :param k: int, k hop value
        :return: K hop sets(including start_node)
        """
        if k < 1:
            raise ValueError("K-hop neighbours value must greater than 1.")

        visited = set()
        current_level = {start_node}

        for _ in range(k):
            next_level = set()  # 下一层的节点
            for node in current_level:
                neighbors = set(await self.neighbors(node))
                next_level.update(neighbors - visited)
            visited.update(next_level)
            current_level = next_level

        return current_level

    async def find_k_hop_neighbors_batch(self, start_nodes: list[str], k: int) -> set:
        nodes_set_list = await asyncio.gather(
            *[self.find_k_hop_neighbors(node, k) for node in start_nodes]
        )
        nodes_list = []
        for node_set in nodes_set_list:
            nodes_list.extend(list(node_set))
        return set(nodes_list)

    async def get_edge_relation_name(self, source_node_id: str, target_node_id: str):
        edge_data = self._graph.edges.get((source_node_id, target_node_id))
        return edge_data.get("relation_name") if edge_data is not None else None

    async def get_edge_relation_name_batch(self, edges: list[tuple[str, str]]):
        relations = await asyncio.gather(
            *[self.get_edge_relation_name(edge[0], edge[1]) for edge in edges]
        )
        return relations

    def clear(self):
        self._graph = nx.Graph()
