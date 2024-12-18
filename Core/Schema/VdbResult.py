from abc import ABC, abstractmethod
import asyncio
class EntityResult(ABC):
    @abstractmethod
    def get_node_data(self):
        pass


class ColbertNodeResult(EntityResult):
    def __init__(self, node_idxs, ranks, scores):
        self.node_idxs = node_idxs
        self.ranks = ranks
        self.scores = scores

    async def get_node_data(self, graph, score = False):
        nodes =  await asyncio.gather(*[graph.get_node_by_index(node_idx) for node_idx in self.node_idxs])
        if score:

            return nodes, [r for r in self.scores]
        else:
            return nodes
    async def get_tree_node_data(self, graph, score = False):
  
        nodes = await asyncio.gather( *[ graph.get_node(node_idx) for node_idx in self.node_idxs])
        if score:

            return nodes, [r for r in self.scores]
        else:
            return nodes

class VectorIndexNodeResult(EntityResult):
    def __init__(self, results):
        self.results = results

    async def get_node_data(self, graph, score = False):

        nodes = await asyncio.gather( *[ graph.get_node(r.metadata["entity_name"]) for r in self.results])
        if score:

            return nodes, [r.score for r in self.results]
        else:
            return nodes
    
    
    async def get_tree_node_data(self, graph, score = False):
  
        nodes = await asyncio.gather( *[ graph.get_node(r.metadata[graph.entity_metakey]) for r in self.results])
        if score:

            return nodes, [r.score for r in self.results]
        else:
            return nodes

class RelationResult(ABC):
    @abstractmethod
    def get_edge_data(self):
        pass

class  VectorIndexEdgeResult(RelationResult):
    def __init__(self, results):
        self.results = results

    async def get_edge_data(self, graph, score = False):

        nodes = await asyncio.gather( *[ graph.get_edge(r.metadata["src_id"], r.metadata["tgt_id"]) for r in self.results])
        if score:

            return nodes, [r.score for r in self.results]
        else:
            return nodes


class SubgraphResult(ABC):
    @abstractmethod
    def get_subgraph_data(self):
        pass


class  VectorIndexSubgraphResult(SubgraphResult):
    def __init__(self, results):
        self.results = results

    async def get_subgraph_data(self,score = False):
        subgraphs_data = list(map(lambda x: {"source_id" : x.metadata["source_id"], "subgraph_content": x.text}, self.results))
        if score:
            return subgraphs_data, [r.score for r in self.results]
        else:
            return subgraphs_data
        
class ColbertEdgeResult(RelationResult):
    def __init__(self, edge_idxs, ranks, scores):
        self.edge_idxs = edge_idxs
        self.ranks = ranks
        self.scores = scores

    async def get_edge_data(self, graph):
        return await asyncio.gather(
            *[(graph.get_edge_by_index(edge_idx), self.scores[idx]) for idx, edge_idx in enumerate(self.edge_idxs)]
        )