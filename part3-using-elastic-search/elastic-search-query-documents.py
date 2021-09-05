from dataclasses import dataclass
from typing import Optional, List

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search

from common import load_json

DATA_IN_PATH = "data/data.json"

documents = load_json(DATA_IN_PATH)

# Connect to elastic seach cluster running on localhost
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])


@dataclass
class RawSearchResult:
    texts: List[str]
    section_titles: List[str]
    article_titles: List[str]


def search(
        query: str,
        sections_to_exclude: Optional[List[str]] = None,
        n_results: int = 50,
        index: str = "",
):
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

    texts = []
    section_titles = []
    article_titles = []

    for h in docs['hits']['hits']:
        texts.append(h["_source"]["text"])
        section_titles.append(h["_source"]["section_title"])
        article_titles.append(h["_source"]["article_title"])

    return RawSearchResult(
        texts=texts,
        section_titles=section_titles,
        article_titles=article_titles,
    )


raw_results = search(
    query="World Health Organization",
    sections_to_exclude=["See also", 'Further reading', 'Data and graphs', 'Medical journals', "External links"]
)





