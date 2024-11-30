import asyncio

from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Graph.TreeGraph import TreeGraph
# from raptor.EmbeddingModels import OpenAIEmbeddingModel

with open("book.txt") as f:
    doc = f.read()

# model = OpenAIEmbeddingModel()
# print(model.create_embedding("Hello"))
# tree_graph = TreeGraph()
# asyncio.run(tree_graph.test())
tree_graph = TreeGraph()
chunks = asyncio.run(tree_graph.chunk_documents(doc))

print(chunks)

asyncio.run(tree_graph._construct_graph(chunks))
answer = asyncio.run(tree_graph.answer_question("What did Scrooge say?"))

print(answer)
# tree_graph._construct_graph(chunks)

# model = get_rag_embedding() 
