
from flask import Flask, render_template, request
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from pymongo import MongoClient
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import dotenv

import os


# openai.api_key = os.getenv('OPENAI_API_KEY')
OAIclient = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

client = MongoClient('mongodb://localhost:27017/')
db = client['kavigai_rag']
collection = db['goal_templates']

from tqdm import tqdm
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer
import numpy as np

from transformers import AutoTokenizer


# # Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
# model

# Retrieve documents from MongoDB
documents = collection.find()
goal_templates_embeddings = []


# Generate embeddings and store them in MongoDB
for doc in documents:
    goal_template_name = doc.get('goal template', '') # avoid error when field is missing
    description = doc['description']  #will be an error when the field is missing
    price = doc['price']
    category = doc.get('category', '')
    
    # Concatenate title and content (or other fields)
    combined_text = f"{goal_template_name}. {description}, {price}. {category}"
    embedding = model.encode(combined_text)
    goal_templates_embeddings.append({'_id': doc['_id'], 'goal_templates_embeddings': embedding.tolist()})

# # Insert embeddings into a separate collection
# if goal_templates_embeddings !exist ont insert 
    # db.drop_collection('goal_templates_embeddings')
goal_templates_embedding_collection = db['goal_templates_embeddings']
######## goal_templates_embedding_collection.insert_many(goal_templates_embeddings)
print("Document embeddings inserted successfully!")



from scipy.spatial.distance import cdist
import numpy as np

def retrieve_documents(query, top_k=3):
    # Generate query embedding
    query_embedding = model.encode(query).reshape(1, -1)

    # Retrieve all embeddings from MongoDB
    embeddings = list(goal_templates_embedding_collection.find())

    # Compute cosine similarity between query embedding and document embeddings
    distances = cdist(query_embedding, np.array([np.array(doc['goal_templates_embeddings']) for doc in embeddings]), 'cosine')

    # Find the top_k closest documents
    closest_docs = np.argsort(distances[0])[:top_k]
    doc_ids = [embeddings[idx]['_id'] for idx in closest_docs]

    # Retrieve full documents from MongoDB
    retrieved_docs = collection.find({"_id": {"$in": doc_ids}})
    
    return list(retrieved_docs)



# Function to retrieve relevant documents from MongoDB using embeddings

def generate_response(query, documents):
    # Prepare the prompt for OpenAI
    prompt = f"Query: {query}\n\n"
    prompt += "Here are some relevant documents:\n"
    
    # Define the System Role
    messages = [
    {"role": "system", "content": f"Find most relevernt documents for the {query}:"}
    ]
    print(f'prompt: {messages}')

 # Add the user's question to the messages as a User Role
    

    for doc in documents:
        messages.append({"role": "user", "content": f"\nGoal Template: {doc['goal template']}\nDescription: {doc['description']}\n"})
        # prompt += f"\nGoal Template: {doc['goal template']}\nDescription: {doc['description']}\n"

    print(f'prompt: {messages}')

    # Send the query to OpenAI
    response = OAIclient.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages
    )

    # res1 = response['choices'][0]['message']['content']
    # print(response['choices'][0]['message']['content'])
    print(response.choices[0].message.content)


    # res1 = response['choices'][0]['message']
    # res2 = response['choices'][0]['message'][0]
    # res3 = response['choices'][0]['message'][0]['content']

    # res2 = response[0].choices[0].message[0].content[0]

    # response.getJSONArray("choices").getJSONObject(0).getString("message")
    # response[0]?.delta?.content || ""
    return response.choices[0].message.content


# Main route for the web interface
# Initialize Flask app
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the query from the user input
        query = request.form['query']

        # Retrieve relevant documents
        documents = retrieve_documents(query)

        # Generate a response using the OpenAI API
        response = generate_response(query, documents)

        return render_template('index.html', query=query, response=response, documents=documents)
    
    return render_template('index.html', query='', response='', documents=[])

if __name__ == '__main__': 
    app.run(debug=True)



