from elasticsearch import Elasticsearch

from common import load_json

DATA_IN_PATH = "data/data.json"

documents = load_json(DATA_IN_PATH)

# Connect to elastic seach cluster running on localhost
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Insert documents in index
for doc in documents:
    res = es.index(index='test-index', body=doc)

# Check that the index was created
print(es.indices.get_alias("*"))

# Get a document count
es.cat.count("*")