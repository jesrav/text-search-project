from transformers import AutoModel, AutoTokenizer
import torch
import faiss
import numpy as np

from common import load_json


# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# model = AutoModel.from_pretrained("distilbert-base-uncased")
#tokenizer = AutoTokenizer.from_pretrained("Maltehb/danish-bert-botxo")
#model = AutoModel.from_pretrained("Maltehb/danish-bert-botxo")
tokenizer = AutoTokenizer.from_pretrained("flax-community/roberta-base-danish")
model = AutoModel.from_pretrained("flax-community/roberta-base-danish")


documents = [
    "I skolen lærer man mange sjove ting. Jeg kan specielt godt lide frikvarter.",
    "Patienten opfører sig vanvittigt. Han bliver ved med, at sige at han er fra fremtiden.",
    "Lebronn James er bedre til fodbold end Jordan",
    "Når månen synger, rejser lynget sig",
]


def encode(document: str) -> torch.Tensor:
    tokens = tokenizer(document, return_tensors='pt')
    vector = model(**tokens)[0].detach().squeeze()
    return torch.mean(vector, dim=0)


encoded_docs = [encode(document) for document in documents]

index = faiss.IndexIDMap(faiss.IndexFlatIP(768)) # the size of our vector space
# index all the documents, we need them as numpy arrays first
index.add_with_ids(
    np.array([t.numpy() for t in encoded_docs]),
    # the IDs will be 0 to len(documents)
    np.array(range(0, len(documents)))
)


def search(query: str, k=1):
  encoded_query = encode(query).unsqueeze(dim=0).numpy()
  top_k = index.search(encoded_query, k)
  scores = top_k[0][0]
  results = [documents[_id] for _id in top_k[1][0]]
  return list(zip(results, scores))


search("Jeg ser sol og stjerner.", k=4)
search("Jeg er syg. Hvor skal jeg hen??.", k=4)
search("Jeg vil gerne spille tennis??.", k=4)