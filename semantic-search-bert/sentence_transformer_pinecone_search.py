import os

from sentence_transformers import SentenceTransformer
import pinecone
from dotenv import load_dotenv, find_dotenv
import numpy as np

from common import load_json

load_dotenv(find_dotenv())

JSON_INPATH = "data/sentences.json"
PINECONE_INDEX_NAME = "sentence-model-index"

sentences = load_json(JSON_INPATH)
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

##################################################
# Setup pinecone index and update index
##################################################
pinecone.init(api_key=os.environ["PINECONE_API_KEY"])

if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(PINECONE_INDEX_NAME)

index = pinecone.Index(PINECONE_INDEX_NAME)

encoded_docs = [model.encode(doc) for doc in sentences]

index.upsert(items=zip(range(len(encoded_docs)), [v for v in encoded_docs]))


##################################################
# Search index
##################################################
def search(query: str, index: pinecone.Index, k=5):
    encoded_query = np.reshape(model.encode(query), (1, -1))
    return index.query(encoded_query, top_k=k)


results = search("How many people have died during Black Death?", index)
for id in results[0].ids:
    print(sentences[int(id)])

# Clean up resources
pinecone.delete_index(PINECONE_INDEX_NAME)