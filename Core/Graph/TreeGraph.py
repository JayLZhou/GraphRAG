from Core.Graph.BaseGraph import BaseGraph
from Core.Storage.JsonKVStorage import JsonKVStorage
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.EntityRelation import Relationship
from Core.Common.Logger import logger
from Core.Common.ContextMixin import ContextMixin
from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Index.VectorIndex import VectorIndex
from Core.Index.Schema import VectorIndexConfig, FAISSIndexConfig
from Core.Prompt.RaptorPrompt import SUMMARIZE, ANSWER_QUESTION
from Core.Community.BaseCommunity import BaseCommunity
from Core.Community.ClusterFactory import get_community_instance
from Core.Community.RaptorClustering import TreeNode

from llama_index.core.embeddings import BaseEmbedding

from typing import Dict, List, Set, Tuple, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

from threading import Lock

from llama_index.core.schema import QueryBundle

import copy
import tiktoken

Embedding = List[float]

def get_text(node_list: List[TreeNode]) -> str:
    """
    Generates a single text string by concatenating the text from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        str: Concatenated text.
    """
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text

def get_node_list(node_dict: Dict[int, TreeNode]) -> List[TreeNode]:
    """
    Converts a dictionary of node indices to a sorted list of nodes.

    Args:
        node_dict (Dict[int, Node]): Dictionary of node indices to nodes.

    Returns:
        List[Node]: Sorted list of nodes.
    """
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list

class Tree:
    """
    Represents the entire hierarchical tree structure.
    """

    def __init__(
        self, all_nodes = None, root_nodes = None, leaf_nodes = None, num_layers = None, layer_to_nodes = None
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes

def get_single_embedding(text: str, embedding_model: BaseEmbedding) -> Embedding:
    embedding = embedding_model._get_text_embedding(text)
    return embedding

def get_embeddings(text: str, embedding_models: Dict[str, BaseEmbedding]) -> Dict[str, Embedding]:
    # return {'EMB1': [0, 0, 0, 0, 0], 'EMB2': [0, 0, 0, 0, 0]}
    embeddings = {
        model_name: model._get_text_embedding(text)
        for model_name, model in embedding_models.items()
    }
    return embeddings

def get_node_list_embeddings(node_list: List[TreeNode], embedding_model: str) -> List:
    """
    Extracts the embeddings of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.
        embedding_model (str): The name of the embedding model to be used.

    Returns:
        List: List of node embeddings.
    """
    return [node.embeddings[embedding_model] for node in node_list]

class TreeGraphConfig():
    reduction_dimension: Optional[int] = 5
    summarization_length: Optional[int] = 100
    num_layers: Optional[int] = 10
    top_k: Optional[int] = 5
    start_layer: Optional[int] = 5
    graph_cluster_params: Optional[dict] = {}
    selection_mode: Optional[str] = "top_k"

    # def __init__(self, reduction_dimension=None, summarization_length = None, num_layers = None):
    #     if (reduction_dimension is None):
    #         reduction_dimension = 5
    #     self.reduction_dimension=reduction_dimension

    #     if summarization_length is None:
    #         summarization_length = 100
    #     self.summarization_length = summarization_length

    #     if num_layers is None:
    #         self.num_layers = 10

class TreeGraph(BaseGraph):

    text_chunks: JsonKVStorage = JsonKVStorage()
    tree_graph: NetworkXStorage = NetworkXStorage()
    tree: Tree = Tree()
    num_layers: int = 0
    embedding_model_name: str = 'EMB'
    embedding_model: BaseEmbedding = get_rag_embedding()
    tree_config: TreeGraphConfig = TreeGraphConfig()
    vector_index_config: VectorIndexConfig = FAISSIndexConfig(persist_path="VectorIndexSave.txt", embed_model = embedding_model)
    graph_cluster_algorithm: BaseCommunity = get_community_instance("raptor")

    def test(self):
        # await self.llm.aask("Hello")
        embedding_models = {self.embedding_model_name: self.embedding_model}
        embeddings = {
            model_name: model.create_embedding("Hello")
            for model_name, model in embedding_models.items()
        }
        # self.ENCODER
        return embeddings

    def embedding_models(self):
        return {self.embedding_model_name: self.embedding_model}

    async def multithreaded_create_leaf_nodes(self, chunks: List[str]) -> Dict[int, TreeNode]:
        """Creates leaf nodes using multithreading from the given list of text chunks.

        Args:
            chunks (List[str]): A list of text chunks to be turned into leaf nodes.

        Returns:
            Dict[int, Node]: A dictionary mapping node indices to the corresponding leaf nodes.
        """

        with ThreadPoolExecutor() as executor:
            future_nodes = {
                executor.submit(self.create_node, index, text): (index, text)
                for index, text in enumerate(chunks)
            }

            leaf_nodes = {}
            for future in as_completed(future_nodes):
                index, node = future.result()
                leaf_nodes[index] = node

        return leaf_nodes

    async def create_node(
        self, index: int, text: str, children_indices: Optional[Set[int]] = None
    ) -> Tuple[int, TreeNode]:
        """Creates a new node with the given index, text, and (optionally) children indices.

        Args:
            index (int): The index of the new node.
            text (str): The text associated with the new node.
            children_indices (Optional[Set[int]]): A set of indices representing the children of the new node.
                If not provided, an empty set will be used.

        Returns:
            Tuple[int, Node]: A tuple containing the index and the newly created node.
        """

        if children_indices is None:
            children_indices = set()

        embeddings = get_embeddings(text, self.embedding_models())

        print("Create node index = {index}, children = {children}".format(index=index, children=children_indices))

        ### insert the node into the tree_graph
        await self.tree_graph.upsert_node(
            node_id=str(index),
            node_data=dict(text=text, index=index, embeddings=embeddings)
        )
        ### inser the edges into the tree_graph
        for children in children_indices:
            await self.tree_graph.upsert_edge(source_node_id=str(index), target_node_id=str(children), edge_data=dict(src_id=str(index), tgt_id=str(children), source_id=None))

        return (index, TreeNode(text, index, children_indices, embeddings))

    async def summarize(self, context, max_tokens=150) -> str:
        """
        Generates a summary of the input context using the specified summarization model.

        Args:
            context (str, optional): The context to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.o

        Returns:
            str: The generated summary.
        """
     
        content = SUMMARIZE.format(context=context)
        return await self.llm.aask(content)

    async def construct_tree(
        self,
        current_level_nodes: Dict[int, TreeNode],
        all_tree_nodes: Dict[int, TreeNode],
        layer_to_nodes: Dict[int, List[TreeNode]],
    ) -> Dict[int, TreeNode]:
        
        self.num_layers = self.tree_config.num_layers
        logger.info("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)

        async def process_cluster(
            cluster, new_level_nodes, next_node_index, summarization_length, lock
        ):
            
            # import pdb
            # pdb.set_trace()
     
            node_texts = get_text(cluster)

            summarized_text = await self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )

            logger.info(
                f"Node Texts Length: {len(self.ENCODER.encode(node_texts))}, Summarized Text Length: {len(self.ENCODER.encode(summarized_text))}"
            )

        

            __, new_parent_node = await self.create_node(
                next_node_index, summarized_text, {node.index for node in cluster}
            )
            with lock:
                new_level_nodes[next_node_index] = new_parent_node

        for layer in range(self.num_layers):

            new_level_nodes = {}

            logger.info(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            # import pdb
            # pdb.set_trace()

            if len(node_list_current_layer) <= self.tree_config.reduction_dimension + 1:
                self.num_layers = layer
                logger.info(
                    f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}"
                )
                break

            clusters = self.graph_cluster_algorithm._clustering_(
                nodes=node_list_current_layer,
                embedding_model_name=self.embedding_model_name,
                reduction_dimension=self.tree_config.reduction_dimension,
                **self.tree_config.graph_cluster_params,
            )


            lock = Lock()

            summarization_length = self.tree_config.summarization_length
            logger.info(f"Summarization Length: {summarization_length}")

        
            for cluster in clusters:
                await process_cluster(
                    cluster,
                    new_level_nodes,
                    next_node_index,
                    summarization_length,
                    lock,
                )
                next_node_index += 1

            # import pdb
            # pdb.set_trace()

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            self.tree = Tree(
                all_tree_nodes,
                layer_to_nodes[layer + 1],
                layer_to_nodes[0],
                layer + 1,
                layer_to_nodes,
            )

        # import pdb
        # pdb.set_trace()
        print(current_level_nodes)
        return current_level_nodes

    async def _construct_graph(self, chunks: dict[str, dict[str, str]], use_multithreading: bool = False):

        
        chunks_list = [value['content'] for key, value in chunks.items()]

        if (use_multithreading):
            leaf_nodes = await self.multithreaded_create_leaf_nodes(chunks_list)
        else:
            leaf_nodes = {}
            for index, text in enumerate(chunks_list):            
                __, node = await self.create_node(index, text)
                leaf_nodes[index] = node

        print(leaf_nodes)

        layer_to_nodes = {0: list(leaf_nodes.values())}

        logger.info(f"Created {len(leaf_nodes)} Leaf Embeddings")

        logger.info("Building All Nodes")

        all_nodes = copy.deepcopy(leaf_nodes)

        root_nodes = await self.construct_tree(all_nodes, all_nodes, layer_to_nodes)

        tree = Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)
        return tree

    async def retrieve_information_collapse_tree(self, query: str, top_k: int, max_tokens: int) -> str:
        """
        Retrieves the most relevant information from the tree based on the query.

        Args:
            query (str): The query text.
            max_tokens (int): The maximum number of tokens.

        Returns:
            str: The context created using the most relevant nodes.
        """

        query_embedding = get_single_embedding(query, self.embedding_model)

        selected_nodes = []

        node_list = get_node_list(self.tree.all_nodes)

        embeddings = get_node_list_embeddings(node_list, self.embedding_model_name)

        index = VectorIndex(self.vector_index_config)
        
        for idx, node in enumerate(node_list):
            await index.upsert_with_embedding(text = node.text, embedding = embeddings[idx], metadata = {'index': idx})
        nodes = await index.retrieval(QueryBundle(query_str = query, embedding = query_embedding), top_k = top_k)
        
        # import pdb
        # pdb.set_trace()
        
        selected_nodes = [self.tree.all_nodes[node.metadata['index']] for node in nodes]

        # distances = distances_from_embeddings(query_embedding, embeddings)

        # indices = indices_of_nearest_neighbors_from_distances(distances)

        # total_tokens = 0
        # for idx in indices[:top_k]:

        #     node = node_list[idx]
        #     node_tokens = len(self.tokenizer.encode(node.text))

        #     if total_tokens + node_tokens > max_tokens:
        #         break

        #     selected_nodes.append(node)
        #     total_tokens += node_tokens

        context = get_text(selected_nodes)
        return selected_nodes, context

    async def retrieve_information(
        self, current_nodes: List[TreeNode], query: str, num_layers: int
    ) -> str:
        """
        Retrieves the most relevant information from the tree based on the query.

        Args:
            current_nodes (List[Node]): A List of the current nodes.
            query (str): The query text.
            num_layers (int): The number of layers to traverse.

        Returns:
            str: The context created using the most relevant nodes.
        """

        query_embedding = get_single_embedding(query, self.embedding_model)

        selected_nodes = []

        node_list = current_nodes

        for layer in range(num_layers + 1):

            embeddings = get_node_list_embeddings(node_list, self.embedding_model_name)

            index = VectorIndex(self.vector_index_config)

            for idx, node in enumerate(node_list):
                await index.upsert_with_embedding(text = node.text, embedding = embeddings[idx], metadata = {'index': idx})

            if self.tree_config.selection_mode == "threshold":
                pass

                # best_indices = [
                #     index for index in indices if distances[index] > self.threshold
                # ]

            elif self.tree_config.selection_mode == "top_k":
                nodes = await index.retrieval(QueryBundle(query_str = query, embedding = query_embedding), top_k = self.tree_config.top_k)

            nodes_to_add = [node_list[node.metadata['index']] for node in nodes]

            selected_nodes.extend(nodes_to_add)

            if layer != num_layers:

                child_nodes = []

                for node in nodes_to_add:
                    child_nodes.extend(node.children)

                # take the unique values
                child_nodes = list(dict.fromkeys(child_nodes))
                node_list = [self.tree.all_nodes[i] for i in child_nodes]

        # import pdb
        # pdb.set_trace()
        context = get_text(selected_nodes)
        return selected_nodes, context

    async def retrieve(
        self,
        query: str,
        start_layer: int = None,
        num_layers: int = None,
        top_k: int = 10, 
        max_tokens: int = 3500,
        collapse_tree: bool = False,
        return_layer_information: bool = False,
    ) -> str:
        """
        Queries the tree and returns the most relevant information.

        Args:
            query (str): The query text.
            start_layer (int): The layer to start from. Defaults to self.start_layer.
            num_layers (int): The number of layers to traverse. Defaults to self.num_layers.
            max_tokens (int): The maximum number of tokens. Defaults to 3500.
            collapse_tree (bool): Whether to retrieve information from all nodes. Defaults to False.

        Returns:
            str: The result of the query.
        """

        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")

        if not isinstance(collapse_tree, bool):
            raise ValueError("collapse_tree must be a boolean")

        # Set defaults
        # import pdb
        # pdb.set_trace()
        start_layer = self.tree_config.start_layer if start_layer is None else start_layer
        num_layers = self.num_layers if num_layers is None else num_layers

        if (start_layer is None):
            start_layer = num_layers
        start_layer = min(start_layer, num_layers)
        start_layer = max(start_layer, 0)

        if not isinstance(start_layer, int) or not (
            0 <= start_layer <= self.num_layers
        ):
            raise ValueError(
                "start_layer must be an integer between 0 and tree.num_layers"
            )

        if not isinstance(num_layers, int) or num_layers < 1:
            raise ValueError("num_layers must be an integer and at least 1")

        if num_layers > (start_layer + 1):
            raise ValueError("num_layers must be less than or equal to start_layer + 1")

        if collapse_tree:
            logger.info(f"Using collapsed_tree")
            selected_nodes, context = await self.retrieve_information_collapse_tree(
                query, top_k, max_tokens
            )
        else:
            layer_nodes = self.tree.layer_to_nodes[start_layer]
            selected_nodes, context = await self.retrieve_information(
                layer_nodes, query, num_layers
            )

        if return_layer_information:

            layer_information = []

            for node in selected_nodes:
                layer_information.append(
                    {
                        "node_index": node.index,
                        "layer_number": self.tree_node_index_to_layer[node.index],
                    }
                )

            return context, layer_information

        return context


    async def answer_question(
        self,
        question,
        top_k: int = 10,
        start_layer: int = None,
        num_layers: int = None,
        max_tokens: int = 3500,
        collapse_tree: bool = False,
        return_layer_information: bool = False,
    ):
        """
        Retrieves information and answers a question using the TreeRetriever instance.

        Args:
            question (str): The question to answer.
            start_layer (int): The layer to start from. Defaults to self.start_layer.
            num_layers (int): The number of layers to traverse. Defaults to self.num_layers.
            max_tokens (int): The maximum number of tokens. Defaults to 3500.
            use_all_information (bool): Whether to retrieve information from all nodes. Defaults to False.

        Returns:
            str: The answer to the question.

        Raises:
            ValueError: If the TreeRetriever instance has not been initialized.
        """
        # if return_layer_information:
        context = await self.retrieve(
            question, start_layer, num_layers, top_k, max_tokens, collapse_tree, return_layer_information
        )

        context = ANSWER_QUESTION.format(context=context, question=question)

        logger.info("Answering question:\n context={context}\n question={question}\n", context=context, question=question)

        answer = await self.llm.aask(context)

        return answer

    def _extract_node(self):
        pass

    def _extract_relationship(self):
        pass

    def _exist_graph(self):
        pass

# tree_graph = TreeGraph()