import numpy as np
import asyncio
from nomic import embed
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding

# nk-ALQs3L7funfp5oHby8uhuocAlmLHkezUteXLg7eBJ8A
class NomicEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query: str) -> Embedding:
        output= embed.text(
                            texts=[query],
                            model='nomic-embed-text-v1.5',
                            task_type='search_query',
                            inference_mode='local',
                          )
        
        return output["embeddings"][0]

    def _get_text_embedding(self, text: str) -> Embedding:
        output= embed.text(
                            texts=[text],
                            model='nomic-embed-text-v1.5',
                            task_type='search_document',
                            inference_mode='local'
                          )
         
        return output["embeddings"][0]
    
    async def _aget_query_embedding(self, query: str) -> Embedding:
        result = await asyncio.to_thread(self._get_query_embedding, query)
        return result
