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



###--- Prompt Query ---###
input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)

print('retrieved_knowledge:')
for chunk, similarity in retrieved_knowledge:
    print(f" - (similarity: {similarity: .2f}) {chunk}")
    
prompt = f'''You are a helpful chatbot. 
Use only the following pieces of context to answer the question. Don't make up any new information:
{'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
'''

###--- Generating Response ---###
stream = ollama.chat(
  model=LANGUAGE_MODEL,
  messages=[
    {'role': 'system', 'content': instruction_prompt},
    {'role': 'user', 'content': input_query},
  ],
  stream=True,
)


print('Chatbot response:')
for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)
