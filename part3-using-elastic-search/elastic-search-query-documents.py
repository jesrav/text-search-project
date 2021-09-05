from dataclasses import dataclass
from typing import Optional, List

import torch
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, util


# Connect to elastic seach cluster running on localhost
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])


@dataclass
class RawSearchResult:
    text: str
    section_title: str
    article_title: str


@dataclass
class SemanticSearchResult(RawSearchResult):
    bert_score: float


def search(
        query: str,
        sections_to_exclude: Optional[List[str]] = None,
        n_results: int = 50,
        index: str = "",
) -> List[RawSearchResult]:
    if not sections_to_exclude:
        sections_to_exclude = []
    query_body = {
        "size": n_results,
        "query": {
            "bool": {
                "should": {
                   "match": {"text": query}
                 },
                "must_not": {
                    "terms": {"section_title.keyword": sections_to_exclude}
                },
            }
        }
    }

    docs = es.search(index=index, body=query_body)

    return [RawSearchResult(
                text=h["_source"]["text"],
                section_title=h["_source"]["section_title"],
                article_title=h["_source"]["article_title"],
            ) for h in docs['hits']['hits']]


def semantic_reranking(
        query: str,
        raw_search_results: List[RawSearchResult],
        embedder: SentenceTransformer,
        top_k: int = 10
):
    corpus_embeddings = embedder.encode([raw.text for raw in raw_search_results], convert_to_tensor=True)
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    reranked_results = (
        util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
    )

    reranked_semantic_search_results = []

    for item in reranked_results:
        idx = item['corpus_id']
        semantic_search_result = SemanticSearchResult(
            bert_score=item['score'],
            article_title=[raw.article_title for raw in raw_search_results][idx],
            section_title=[raw.section_title for raw in raw_search_results][idx],
            text=[raw.text for raw in raw_search_results][idx],
        )

        reranked_semantic_search_results.append(semantic_search_result)

    return reranked_semantic_search_results


query = "What diseases spread fast and in the air?"
raw_results = search(
    query=query,
    sections_to_exclude=["See also", 'Further reading', 'Data and graphs', 'Medical journals', "External links"]
)

embedder = SentenceTransformer("distilbert-base-nli-stsb-mean-tokens")
semantic_results = semantic_reranking(
    query,
    raw_results,
    embedder,
)

[(r.article_title, r.section_title)  for r in semantic_results]

[(r.article_title, r.section_title)  for r in raw_results][:10]



