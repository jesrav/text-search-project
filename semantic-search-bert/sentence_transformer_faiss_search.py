from typing import List
import pickle

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from common import load_json

JSON_INPATH = "data/sentences.json"
FAISS_INDEX_OUTPATH = "data/sentence_transformer_faiss_index.pickle"

sentences = load_json(JSON_INPATH)
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')


class SentenceTransformerFaissIndexSearch:
    """Toy implementation of faiss index search with sentence transformer"""
    def __init__(self, model: SentenceTransformer, documents: List[str]):
        self.model = model
        self.documents = documents
        self.embeded_docs = [self.model.encode(doc) for doc in documents]
        self.vector_space_size = 768
        self.faiss_index = self._build_faiss_index(self.embeded_docs)

    def _build_faiss_index(self, encoded_docs):
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.vector_space_size))
        index.add_with_ids(
            np.array([v for v in encoded_docs]),
            np.array(range(len(encoded_docs))),
        )
        return index

    def search(self, query: str, k=5):
      encoded_query = np.reshape(self.model.encode(query), (1, -1))
      top_k = self.faiss_index.search(encoded_query, k)
      scores = top_k[0][0]
      results = [self.documents[_id] for _id in top_k[1][0]]
      return list(zip(results, scores))


faiss_search = SentenceTransformerFaissIndexSearch(
    model=model,
    documents=load_json(JSON_INPATH)
)

# Testing it out
faiss_search.search(100*"How many people have died during Black Death?")
faiss_search.search("Viruses in nanotechnology")

# Serialize faiss index
with open(FAISS_INDEX_OUTPATH, "wb") as f:
    pickle.dump(faiss_search.faiss_index, f)