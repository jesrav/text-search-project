from itertools import chain
from typing import List
from operator import itemgetter

from add_tokens import add_tokens
from common import load_json, write_json
from schemas import DocumentWithTokens, DocumentsWithTFIDF, Document

import click
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

JSON_DATA_INPATH = "../data/data_with_tokens.json"
JSON_DATA_OUTPATH = "../data/data_with_tokens_and_tfidf.json"
JSON_TRANSFORMER_OUTPATH = "../data/seriealized_tfidf_transformer.json"


class TFIDFTransformer:
    """Transformer to get TFIDF vectors."""
    def __init__(self, inverse_document_frequencies=None, corpus_vocabulary = None):
        self.inverse_document_frequencies = inverse_document_frequencies
        self.corpus_vocabulary = corpus_vocabulary

    @staticmethod
    def get_term_frequency(token: str, document: DocumentWithTokens):
        return document.tokenized_text.count(token) / len(document.tokenized_text)

    @staticmethod
    def get_inverse_document_frequency(
            token: str, documents: List[DocumentWithTokens]
    ):
        n_documents_containing_token = len(
            [doc for doc in documents if token in doc.tokenized_text]
        )
        n_documents = len(documents)
        return np.log(n_documents / n_documents_containing_token)

    def fit(self, documents: List[DocumentWithTokens]) -> None:
        self.corpus_vocabulary = list(set(
            chain(*[doc.tokenized_text for doc in documents])
        ))
        self.inverse_document_frequencies = {
            token: self.get_inverse_document_frequency(token, documents)
            for token in self.corpus_vocabulary
        }

    def get_tfidfs(
            self,
            document: DocumentWithTokens,
    ) -> List[float]:
        tfidf_dict = {
            token: (
                self.get_term_frequency(token, document)
                * self.inverse_document_frequencies[token]
            )
            for token in self.corpus_vocabulary
        }
        return list(tfidf_dict.values())

    def transform(self, document: DocumentWithTokens) -> DocumentsWithTFIDF:
        tfidfs = self.get_tfidfs(document)
        return DocumentsWithTFIDF(
            title=document.title,
            text=document.text,
            url=document.url,
            tokenized_text=document.tokenized_text,
            tfidfs=tfidfs
        )

    def to_dict(self):
        return {
            "inverse_document_frequencies": self.inverse_document_frequencies,
            "corpus_vocabulary": self.corpus_vocabulary,
        }

    @classmethod
    def from_dict(cls, d):
        return TFIDFTransformer(**d)


def get_index_of_n_highest_values(l: list, n: int) -> list[int]:
    return sorted(range(len(l)), key=lambda x: l[x])[-n:][::-1]


def tfidf_search(document: DocumentsWithTFIDF, document_db: [DocumentsWithTFIDF], n: int = 5):
    """Find the n most similar documents using cosine similarity."""
    cosine_similarities = [
        float(cosine_similarity(
            np.array(document.tfidfs).reshape(1, -1),
            np.array(doc.tfidfs).reshape(1, -1)
        ).squeeze())
        for doc in document_db
    ]
    index_of_n_highest_values = get_index_of_n_highest_values(cosine_similarities, n)
    return itemgetter(*index_of_n_highest_values)(document_db)


@click.command()
@click.option('--search', is_flag=True, help='Search for documents.')
def main(search):
    wiki_articles = [
        DocumentWithTokens(**wiki_data) for wiki_data in load_json(JSON_DATA_INPATH)
    ]

    transformer = TFIDFTransformer()

    transformer.fit(wiki_articles)

    wiki_articles_with_tfidf = [
        transformer.transform(wiki_article) for wiki_article in wiki_articles
    ]

    if search:
        search_text = input("Enter seacrh text: ")
        search_doc = Document(text=search_text)
        search_doc_with_tokens = add_tokens(documents=[search_doc])[0]
        search_doc_with_tfidf = transformer.transform(document=search_doc_with_tokens)
        relevant_docs = tfidf_search(search_doc_with_tfidf, wiki_articles_with_tfidf)
        print("Titles of top five most similar articles in database:")
        for doc in relevant_docs:
            print(doc.title)

    else:
        write_json(
            object=[wiki_article.dict() for wiki_article in wiki_articles_with_tfidf],
            json_path=JSON_DATA_OUTPATH
        )
        write_json(
            object=transformer.to_dict(),
            json_path=JSON_TRANSFORMER_OUTPATH
        )


if __name__ == "__main__":
    main()