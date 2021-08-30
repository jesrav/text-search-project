from typing import List, Optional

from pydantic import BaseModel

tokens = List[str]


class Document(BaseModel):
    """Class to hold a document"""
    title: Optional[str]
    text: str
    url: Optional[str]


class DocumentWithTokens(Document):
    """Class to hold a document with tokens"""
    tokenized_text: tokens


class DocumentsWithTFIDF(DocumentWithTokens):
    """Class to hold a document with tfidfs"""
    tfidfs: List[float]

