from typing import List, Optional, Dict

from pydantic import BaseModel
from numpy import array

tokens = List[str]


class RawWikiArticle(BaseModel):
    """Class to hold a raw wiki article"""
    title: str
    text: str
    url: str


class WikiArticleWithToken(RawWikiArticle):
    """Class to hold a wiki article with tokens"""
    tokenized_text: tokens


class WikiArticleWithTFIDF(WikiArticleWithToken):
    """Class to hold a wiki article with tfidfs"""
    tfidfs: Dict[str, int]