import asyncio
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from huggingface_hub import login

# Token for huggingface_hub access.
# Goto https://huggingface.co/settings/tokens and create one
login(token="")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# Create the llm instance
llm_model=Ollama(model="llama3.2", request_timeout=360.0)

documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents=documents, 
                                        embed_model=embed_model)

query_engine = index.as_query_engine(llm=llm_model)

async def search_documents(query: str) -> str:
    response = await query_engine.aquery(query)
    return str(response)

agent = AgentWorkflow.from_tools_or_functions(
    [search_documents],
    llm=llm_model,
    system_prompt="""You are a helpful assistant that can search through documents to answer questions.""",
)

async def main():
    # Run the agent
    response = await agent.run("Who is Amila Rathnayake and what languages that he knows")
    print(str(response))

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())