import json
from typing import List, Optional

from pydantic import BaseModel
import spacy

JSON_DATA_INPATH = "data/data.json"
JSON_DATA_OUTPATH = "data/data_processed.json"

nlp = spacy.load("en_core_web_sm")


class WikiArticle(BaseModel):
    """Class to hold a wii article"""
    title: str
    text: str
    url: str
    tokenized_text: Optional[List[str]]


def load_wiki_articles() -> List[WikiArticle]:
    """Load wiki data from json file.

    Returns as a list of dictionaries, one per wiki article.
    """
    with open(JSON_DATA_INPATH, "r") as f:
        wiki_articles = json.load(f)
    return [WikiArticle(**wiki_article) for wiki_article in wiki_articles]


def get_tokens(text: str) -> List[str]:
    """get tokenized text from text string"""
    text_lowercase = text.lower()

    doc = nlp(text_lowercase)
    return [
        token.lemma_ for token in doc
        if not (token.is_punct | token.is_space | token.is_stop)
    ]


def preprocess_wiki_articles(wiki_articles: List[WikiArticle]) -> List[WikiArticle]:
    return [WikiArticle(
                title=wiki_article.title,
                text=wiki_article.text,
                url=wiki_article.url,
                tokenized_text=get_tokens(wiki_article.text)
            ) for wiki_article in wiki_articles]


if __name__ == "__main__":
    wiki_articles = load_wiki_articles()
    wiki_articles_with_tokens = preprocess_wiki_articles(wiki_articles)

    with open(JSON_DATA_OUTPATH, "w") as f:
        json.dump(
            [wiki_article.dict() for wiki_article in wiki_articles_with_tokens],
            f
        )