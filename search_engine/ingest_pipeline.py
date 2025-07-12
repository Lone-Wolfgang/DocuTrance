from docutrance.index import (
    build_segment_dataframe, 
    build_wikipedia_index,
    index_documents
)
from configs.index import (
    PATHS,
    INDEX_SETTINGS,
    DOCUMENT_INDEX_MAPPINGS,
    DOCUMENT_INDEX_NAME,
    ENCODER,
    CLIENT,
    LEMMATIZER,
    SEGMENT_INDEX_NAME,
    SEGMENT_INDEX_MAPPINGS
)

urls = PATHS['urls'].read_text().splitlines()


document_df = build_wikipedia_index(
    urls,
    LEMMATIZER,
    ENCODER,
    PATHS['output']
)

index_documents(
    document_df,
    CLIENT,
    DOCUMENT_INDEX_NAME,
    INDEX_SETTINGS,
    DOCUMENT_INDEX_MAPPINGS,
    'document_id'
)

segment_df = build_segment_dataframe(
    document_df,
    LEMMATIZER,
    ENCODER
)

index_documents(
    segment_df,
    CLIENT,
    SEGMENT_INDEX_NAME,
    INDEX_SETTINGS,
    SEGMENT_INDEX_MAPPINGS,
    'segment_id'
)