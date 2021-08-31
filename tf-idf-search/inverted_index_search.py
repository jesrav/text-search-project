from add_tokens import add_tokens
from add_tdidf_vectors import TFIDFTransformer, tfidf_search
from common import load_json
from schemas import Document, DocumentWithTokens, DocumentsWithTFIDF

import click

JSON_DATA_INPATH = "../data/data_with_tokens_and_tfidf.json"
JSON_TRANSFORMER_INPATH = "../data/seriealized_tfidf_transformer.json"

tfidf_transformer = TFIDFTransformer().from_dict(load_json(JSON_TRANSFORMER_INPATH))


def search(search_text, inverse_index):
    search_doc = Document(text=search_text)
    search_doc = add_tokens(documents=[search_doc])[0]
    search_doc = tfidf_transformer.transform(document=search_doc)

    relevant_doc

@click.command()
@click.option('--search', is_flag=True, help='Search for documents.')
def main(search):

    wiki_articles = [
        DocumentsWithTFIDF(**wiki_data) for wiki_data in load_json(JSON_DATA_INPATH)
    ]


    inverted_index = {}
    for i, token in enumerate(tfidf_transformer.corpus_vocabulary):
        inverted_index[token] = []
        for doc in wiki_articles:
            if doc.tfidfs[i] != 0:
                inverted_index[token].append((doc.title, doc.tfidfs[i]))


    search_text = input("Enter seacrh text: ")

    wiki_articles_containing_tokens =
    relevant_docs = tfidf_search(search_doc, wiki_articles)
    print("Titles of top five most similar articles in database:")
    for doc in relevant_docs:
        print(doc.title)


if __name__ == "__main__":
    main()