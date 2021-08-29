from itertools import chain
from typing import List

from common import load_json, write_json
from schemas import WikiArticleWithToken, WikiArticleWithTFIDF

import click
from sklearn.metrics.pairwise import cosine_similarity

JSON_DATA_INPATH = "../data/data_with_tokens.json"
JSON_DATA_OUTPATH = "../data/data_with_tokens_and_tfidf.json"
JSON_TRANSFORMER_OUTPATH = "../data/seriealized_tfidf_transformer.json"


class TFIDFTransformer:

    def __init__(self, inverse_document_frequencies=None, corpus_vocabulary = None):
        self.inverse_document_frequencies = inverse_document_frequencies
        self.corpus_vocabulary = corpus_vocabulary

    @staticmethod
    def get_term_frequency(token: str, document: WikiArticleWithToken):
        return document.tokenized_text.count(token) / len(document.tokenized_text)

    @staticmethod
    def get_inverse_document_frequency(
            token: str, documents: List[WikiArticleWithToken]
    ):
        n_documents_containing_token = len(
            [doc for doc in documents if token in doc.tokenized_text]
        )
        return n_documents_containing_token / len(documents)

    def fit(self, documents: List[WikiArticleWithToken]) -> None:
        self.corpus_vocabulary = list(set(
            chain(*[doc.tokenized_text for doc in documents])
        ))
        self.inverse_document_frequencies = {
            token: self.get_inverse_document_frequency(token, documents)
            for token in self.corpus_vocabulary
        }

    def get_tfidfs(
            self,
            document: WikiArticleWithToken,
    ) -> List[float]:
        tfidf_dict = {
            token: (
                self.get_term_frequency(token, document)
                * self.inverse_document_frequencies[token]
            )
            for token in self.corpus_vocabulary
        }
        return list(tfidf_dict.values())

    def transform(self, document: WikiArticleWithToken) -> WikiArticleWithTFIDF:
        tfidfs = self.get_tfidfs(document)
        return WikiArticleWithTFIDF(
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


@click.command()
@click.option('--search', is_flag=True, help='Search for documents.')
def main(search):
    wiki_articles = [
        WikiArticleWithToken(**wiki_data) for wiki_data in load_json(JSON_DATA_INPATH)
    ]

    transformer = TFIDFTransformer()

    transformer.fit(wiki_articles)

    wiki_articles_with_tfidf = [
        transformer.transform(wiki_article) for wiki_article in wiki_articles
    ]

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