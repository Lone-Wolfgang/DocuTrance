from sentence_transformers import SentenceTransformer
import spacy
from opensearchpy import OpenSearch
from docutrance.util import lemmatize
from pathlib import Path

PATHS = {
    "urls": Path('data/links/countries.txt'),
    "output": Path('data\tables\wikipedia_countries.parquet')
}
ENCODER = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
DIMENSIONS = ENCODER.get_sentence_embedding_dimension()

LEMMATIZER = spacy.load('en_core_web_sm')
CLIENT = OpenSearch(hosts=[{'host': 'localhost', 'port': 9200}])


INDEX_SETTINGS = {
    "index.knn": True,
    "number_of_shards": 1,
    "number_of_replicas": 0
}

DOCUMENT_INDEX_NAME = "wikipedia-documents"

DOCUMENT_INDEX_MAPPINGS = {
    "properties": {
        "url": {"type": "keyword"},
        "body": {"type": "text"},
        "body_lemmatized": {"type": "text"},
        "title": {"type": "text"},
        "title_embedding": {
            "type": "knn_vector",
            "index": True,
            "similarity": "l2_norm",
            "dimension": DIMENSIONS,
            "method": {
                "engine": "lucene",
                "space_type": "l2",
                "name": "hnsw",
                "parameters": {}
                }
        }
    }
}


SEGMENT_INDEX_NAME = "wikipedia-segments"

SEGMENT_INDEX_MAPPINGS = {
    "properties": {
        "document_id": {"type": "keyword"},
        "segment": {"type": "text"},
        "segment_embedding": {
            "type": "knn_vector",
            "index": True,
            "similarity": "l2_norm",
            "dimension": DIMENSIONS,
            "method": {
                "engine": "lucene",
                "space_type": "l2",
                "name": "hnsw",
                "parameters": {}
            }
        }
    }
}
