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

from huggingface_hub import login

from light_rag import LightRAG

async def main():
    rag = LightRAG("user1_rag")
    rag.LoadDocuments("./docs")
    response = await rag.Query("What are the two movies mentioned?")

    print("Answer   ============ \n{}".format(str(response)))
    print("Sources  ============ \n{}".format(response.get_formatted_sources()))

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())