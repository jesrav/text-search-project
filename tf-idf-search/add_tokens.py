from typing import List
import spacy

from common import load_json, write_json
from schemas import RawWikiArticle, WikiArticleWithToken

JSON_DATA_INPATH = "../data/data.json"
JSON_DATA_OUTPATH = "../data/data_with_tokens.json"

nlp = spacy.load("en_core_web_sm")


def get_tokens(text: str) -> List[str]:
    """get tokenized text from text string"""
    text_lowercase = text.lower()

    doc = nlp(text_lowercase)
    return [
        token.lemma_ for token in doc
        if not (token.is_punct | token.is_space | token.is_stop)
    ]


def add_tokens(wiki_articles: List[RawWikiArticle]) -> List[WikiArticleWithToken]:
    return [WikiArticleWithToken(
                title=wiki_article.title,
                text=wiki_article.text,
                url=wiki_article.url,
                tokenized_text=get_tokens(wiki_article.text)
            ) for wiki_article in wiki_articles]


if __name__ == "__main__":
    wiki_articles = [
        RawWikiArticle(**wiki_article) for wiki_article in load_json(JSON_DATA_INPATH)
    ]
    wiki_articles_with_tokens = add_tokens(wiki_articles)
    write_json(
        object=[wiki_article.dict() for wiki_article in wiki_articles_with_tokens],
        json_path=JSON_DATA_OUTPATH
    )