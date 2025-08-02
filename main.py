###--- A Simple RAG Demo ---###
import ollama


###--- Loading the dataset ---###
dataset = []
with open('cat-facts.txt', 'r') as file: 
    dataset = file.readlines()
    print(f"Loaded {len(dataset)} entries")
    
    
###--- Implementing the vector database ---###
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

vector_db = []

def add_chunk(chunk):
    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    vector_db.append((chunk, embedding))
    
for i, chunk in enumerate(dataset):
    add_chunk(chunk)
    print(f"Added chunk {i+1}/{len(dataset)} to the database")
    