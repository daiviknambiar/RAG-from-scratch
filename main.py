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


###--- Implement cosine similarity to return most relevant chunks ---###
def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a,b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    return (dot_product) / (norm_a * norm_b)


###--- Implement retrieval function ---###)
def retrieve(query, top_n=3):
    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]

    similarities = []
    for chunk, embedding in vector_db: 
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]
