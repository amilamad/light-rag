import asyncio
from llama_index.core.agent.workflow import AgentWorkflow

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

from extensions.nomic_embedding import NomicEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.llms.mock import MockLLM
from llama_index.llms.ollama import Ollama

from huggingface_hub import login

# nk-ALQs3L7funfp5oHby8uhuocAlmLHkezUteXLg7eBJ8A
print("RAG starting")

# Load custom embedding 
embed_model = NomicEmbedding();
#embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", local_files_only=True)
print("Loaded custom embedding model")

documents = SimpleDirectoryReader("./docs").load_data()
for doc in documents:
    print(doc)

splitter = SentenceSplitter(chunk_size=200)  # Experiment with chunk size
nodes = splitter.get_nodes_from_documents(documents)
print("Number of nodes {}".format(len(nodes)))

index = VectorStoreIndex(nodes=nodes,
                        embed_model=embed_model)

# Create the llm instance
llm_model=Ollama(model="llama3.2", request_timeout=360.0)
mock_llm_model=MockLLM()
print("Loaded LLM model")

query_engine = index.as_query_engine(similarity_top_k=1, llm=mock_llm_model)

async def search_documents(query: str) -> str:
    print("Searching document index for query {}".format(query))
    response = await query_engine.aquery(query)
    return str(response)

agent = AgentWorkflow.from_tools_or_functions(
    [search_documents],
    llm=llm_model,
    system_prompt="""You are a helpful assistant that can route questions to search_documents and get the answer""",
)

async def main():
    # Run the agent
    response = await search_documents("Amila?")
    print(str(response))

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())