import asyncio
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from extensions.nomic_embedding import NomicEmbedding

from huggingface_hub import login

from nomic import embed
import numpy as np

# nk-ALQs3L7funfp5oHby8uhuocAlmLHkezUteXLg7eBJ8A
print("RAG starting")
embed_model = NomicEmbedding();

print("Loaded custom embedding model")

documents = SimpleDirectoryReader("./docs").load_data()
for doc in documents:
    print(doc)

index = VectorStoreIndex.from_documents(documents=documents, 
                                        embed_model=embed_model)

# Create the llm instance
llm_model=Ollama(model="llama3.2", request_timeout=360.0)
print("Loaded LLM model")

query_engine = index.as_query_engine(llm=llm_model)

async def search_documents(query: str) -> str:
    print("Searching document index for query {}".format(query))
    response = await query_engine.aquery(query)
    return str(response)

agent = AgentWorkflow.from_tools_or_functions(
    [search_documents],
    llm=llm_model,
    system_prompt="""You are a helpful assistant that can search through documents to answer questions.""",
)

async def main():
    # Run the agent
    response = await agent.run("Who is Amila Rathnayake")
    print(str(response))

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())