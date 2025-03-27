import asyncio

from light_rag import LightRAG

async def main():
    rag = LightRAG("user1_rag")
    rag.load_documents("./docs")
    response = await rag.query("What are the two movies mentioned?")

    print("Answer:   \n{}".format(str(response)))
    print("Sources:  \n{}".format(response.get_formatted_sources(length=1000)))

# Run the agent
if __name__ == "__main__":
    asyncio.run(main())