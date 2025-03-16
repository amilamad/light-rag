import asyncio
from llama_index.core.agent.workflow import AgentWorkflow

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from extensions.nomic_embedding import NomicEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.llms.mock import MockLLM
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb

class LightRAG:
    def __init__(self, id: str):
        self.id = id
                            
        self.llm_model=Ollama(model="llama3.2", request_timeout=360.0)
        print("Loaded LLM model")

        self.embed_model = NomicEmbedding()
        print("Loaded custom embedding model")
        
    def load_documents(self, path:str):

        # Initialize the Chroma client
        chroma_client = chromadb.PersistentClient(path="./chroma_db")

        collection_name = self.id
        collections = chroma_client.list_collections()

        self.index = None
        if not collection_name in collections:
            print("Collection {} was not found. Creating ... ".format(collection_name))

            documents = SimpleDirectoryReader(path).load_data()
            
            # Choose better chlunk size for breaking the text. 
            # If too big context text match size will be too big.
            splitter = SentenceSplitter(chunk_size=200) 
            nodes = splitter.get_nodes_from_documents(documents)

            # Create collection
            chroma_collection = chroma_client.create_collection(collection_name)

            # LlamaIndex wrapper for Chroma db
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            self.index = VectorStoreIndex(nodes=nodes,
                                    embed_model=self.embed_model,
                                    storage_context=storage_context)
        else:
            chroma_collection = chroma_client.get_collection(collection_name)

            print("Collection {} is already in the data base".format(collection_name))

            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store,
                                                    embed_model=self.embed_model)

    async def query(self, query:str):
        self.query_engine = self.index.as_query_engine(similarity_top_k=4, llm=self.llm_model)
        print("Searching document index for query {}".format(query))
        return await self.query_engine.aquery(query)

