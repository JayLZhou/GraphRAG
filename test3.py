from Core.Graph.ERGraph import ERGraph
import json
import asyncio
if __name__ == "__main__":
    
    
    graph = ERGraph()
    corpus = json.load(open('./hotpotqa_corpus.json', 'r'))
    keys = list(corpus.keys())
    retrieval_corpus = [{'id': i, 'content': key + '\n' + ''.join(corpus[key])} for i, key in enumerate(keys)]
    import pdb
    pdb.set_trace()
    chunks = asyncio.run(graph.chunk_documents(retrieval_corpus, is_chunked=True))

    asyncio.run(graph._construct_graph(chunks))

    
    asyncio.run(graph.query("who are you"))
   