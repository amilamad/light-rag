import asyncio
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.ollama import Ollama

# Define a simple calculator tool
def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b


# Create an agent workflow with our calculator tool
agent = FunctionAgent(
    name="Agent",
    description="Useful for multiplying two numbers",
    llm=Ollama(model="llama3.2", request_timeout=360.0),
    system_prompt="You are a helpful assistant that can multiply two numbers.",
)


async def main():
    # Run the agent
    response = await agent.run("What is the capital of Sri Lanka")
    print(str(response))


# Run the agent
if __name__ == "__main__":
    asyncio.run(main())