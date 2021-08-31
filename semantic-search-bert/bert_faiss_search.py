from typing import List

from transformers import AutoModel, AutoTokenizer
import torch
import faiss
import numpy as np

from common import load_json

JSON_INPATH = "data/sentences.json"

sentences = load_json(JSON_INPATH)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")


class FaissIndexSearch:
    """Toy implementation of faiss index search with word embeddings"""
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.embeded_docs = [self._encode(doc) for doc in documents]
        self.vector_space_size = 768
        self.faiss_index = self._build_faiss_index(self.embeded_docs)

    @staticmethod
    def _encode(document: str) -> torch.Tensor:
        tokens = tokenizer(document, return_tensors='pt')
        vector = model(**tokens)[0].detach().squeeze()
        return torch.mean(vector, dim=0)

    def _build_faiss_index(self, encoded_docs):
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.vector_space_size))
        index.add_with_ids(
            np.array([t.numpy() for t in encoded_docs]),
            np.array(range(len(encoded_docs))),
        )
        return index

    def search(self, query: str, k=5):
      encoded_query = self._encode(query).unsqueeze(dim=0).numpy()
      top_k = self.faiss_index.search(encoded_query, k)
      scores = top_k[0][0]
      results = [self.documents[_id] for _id in top_k[1][0]]
      return list(zip(results, scores))


faiss_search = FaissIndexSearch(documents=load_json(JSON_INPATH))

# Testing it out
faiss_search.search("How many people have died during Black Death?")
faiss_search.search("Viruses in nanotechnology")
