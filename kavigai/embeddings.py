import numpy as np
import pandas as pd


df = pd.read_csv('goal_templates.csv')


# Clean the documents. 
# In this case it is remove spaces after commas and lowercase the header
df.columns = [column.strip().lower() for column in df.columns]
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# ##### Connect to MongoDB
# 
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['kavigai_rag']
collection = db['goal_templates']

# #### Convert documents in DataFrame to Dictionary and Insert them into the kavigai_rag vector databse
docu_dict = df.to_dict('records')
collection.insert_many(docu_dict);

# #### Create embeddings for the documents and Insert them into the kavigai_rag vector databse
from sentence_transformers import SentenceTransformer

# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Retrieve documents from vector databse
documents = collection.find()
goal_templates_embeddings = []

# Generate embeddings and store them in vector databse
for doc in documents:
    goal_template_name = doc.get('goal_template', '') # use get() to avoid error when field is missing
    description = doc['description']  # will be an error when the field is missing
    price = doc.get('price', '')
    category = doc.get('category', '')
    
    # Concatenate title and content (or other fields)
    combined_text = f"{goal_template_name}. {description}, {price}. {category}"
    embedding = model.encode(combined_text)
    goal_templates_embeddings.append({'_id': doc['_id'], 'goal_templates_embeddings': embedding.tolist()})

# Insert embeddings into a separate collection
goal_templates_embedding_collection = db['goal_templates_embeddings']
# goal_templates_embedding_collection.insert_many(goal_templates_embeddings)
print("\nDocument embeddings inserted successfully! {documents}")
