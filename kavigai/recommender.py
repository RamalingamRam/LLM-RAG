from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['kavigai_rag']
collection = db['goal_templates']

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



import os
import openai
from openai import OpenAI

OAIclient = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


# Function to retrieve relevant documents from MongoDB using embeddings
def generate_response(query, documents):
    # Prepare the prompt for OpenAI
    prompt = f"Query: {query}\n\n"
    prompt += "Here are some relevant documents:\n"
    
    # Define the System Role
    messages = [
    {"role": "system", "content": f"Find most relevernt documents for the {query}:"}
    ]
   
    # Add the user's query to the messages as a User Role
    for doc in documents:
        messages.append({"role": "user", "content": f"\nGoal Template: {doc['goal template']}\nDescription: {doc['description']}\n"})
 
    # Send the query to OpenAI
    response = OAIclient.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=messages
    )
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
