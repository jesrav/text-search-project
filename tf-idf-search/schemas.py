from typing import List, Optional

from pydantic import BaseModel


class RawWikiArticle(BaseModel):
    """Class to hold a raw wiki article"""
    title: str
    text: str
    url: str


class WikiArticleWithToken(RawWikiArticle):
    """Class to hold a wiki article with tokens"""
    tokenized_text: Optional[List[str]]