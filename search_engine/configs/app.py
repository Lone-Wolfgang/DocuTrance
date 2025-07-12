from opensearchpy import OpenSearch
import pandas as pd
from sentence_transformers import SentenceTransformer
import spacy

K=500


#Controls the text displayed in the user interface.
TAB_TITLE = "Document Search"
PAGE_TITLE = "📄 Browse some therapists from Wikipedia"
USER_PROMPT = ("Enter your search query:", "Search documents. . .")
DOCUMENTS = pd.read_parquet('data/tables/wikipedia_countries.parquet')

#Post processing weighting for document, segment, and title search.
WEIGHTS = [0.4, 0.4, 0.2]
CLIENT = OpenSearch(hosts=[{'host': 'localhost', 'port': 9200}])
ENCODER = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
LEMMATIZER = spacy.load('en_core_web_sm')


DOCUMENT_INDEX_NAME = "wikipedia-documents"
SEGMENT_INDEX_NAME = "wikipedia-segments"


DOCUMENT_SEARCH_KWARGS = {
    "type_": "bool",
    "subqueries": [
        {
            "subquery_type": "multi_match",
            "input_type": "stripped",
            "fields": ["title", "body"]
        },
        {
            "subquery_type": "match_phrase",
            "input_type": "lemmatized",
            "field": "body_lemmatized"
        }
    ],
    "highlight": {
        "pre_tags": ["**"],
        "post_tags": ["**"],
        "fields": {
            "body": {
            "fragment_size": 300,
            "number_of_fragments": 3
            }
        }
    }
}

SEGMENT_SEARCH_KWARGS = {
    "type_": "bool",
    "subqueries": [
        {
            "subquery_type": "knn",
            "input_type": "embedding",
            "field": "segment_embedding",
            "k":K
        }
    ],
}

TITLE_SEARCH_KWARGS ={
    "type_": "bool",
    "subqueries": [
        {
            "subquery_type": "knn",
            "input_type": "embedding",
            "field": "title_embedding",
            "k":K
        }
    ],
}

#zip jobs in order kwargs, index, column_map, agg_map, weight

JOBS = list(zip(
    [DOCUMENT_SEARCH_KWARGS, SEGMENT_SEARCH_KWARGS, TITLE_SEARCH_KWARGS], # search kwargs
    [DOCUMENT_INDEX_NAME, SEGMENT_INDEX_NAME, DOCUMENT_INDEX_NAME], #index names
    [{'_id': 'document_id', 'body_highlight': 'keyword_highlight'}, {'segment': 'semantic_highlight'}, {'_id': 'document_id'}], #new column name mappings,
    [{'_score': 'sum', 'keyword_highlight': 'sum'}, {'_score': 'sum', 'semantic_highlight': lambda x: list(x)}, {'_score': 'sum'}], #column aggregation mappings
    WEIGHTS 
))


#Controls the maximum number of documents to retrieve from the
#archive, and the number of results per page.
RESULT_SIZE = 200
PAGE_SIZE = 10