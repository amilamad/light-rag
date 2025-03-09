from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

import torch

print(torch.version.cuda) 

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA on {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

# Token for huggingface_hub access.
# Goto https://huggingface.co/settings/tokens and create one
login(token="")

print("Loading model ....")
# Loading a local model that in the disk 
# Defining the model without a path will make it download from hugging face site. 
# This is whta the above login token is used for,
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load and index the PDF
print("Loading documents ....")
documents = SimpleDirectoryReader("./docs").load_data()
print("Documents loaded ....")

# Will use custom embeding model instead using the default OpenAI embedding. Because this system is totally private to organization.
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
index = VectorStoreIndex.from_documents(documents=documents, 
                                        embed_model=embed_model)

# Create a query engine
query_engine = index.as_query_engine(llm=model)
