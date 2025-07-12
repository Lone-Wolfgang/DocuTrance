# Build your Own Search Engine

Follow this procedure to build a search engine that indexes a collection of Wikipedia pages.

We begin by installing Docker, configuring an OpenSearch node, and setting up a Python environment. Once the setup is complete, the Python script `ingest_pipeline.py` takes a list of Wikipedia URLs and prepares the data for both keyword and semantic search.

After the index is built, users can explore the content and extract highlights using `app.py`.

## Installation

To install Docker, follow the official guide:  
[https://docs.docker.com/get-started/get-docker/](https://docs.docker.com/get-started/get-docker/)

Once Docker is installed and running, you can clone the repository and set up your Python environment. This project was developed using Python 3.10 and has not been tested with other versions.

```bash
#Setup a virtual environemnt with Python 3.10
conda create python=3.10 {venv}

# Activate your environemnt and install dependencies
conda activate {venv}
pip install docutrance
python -m spacy download en_core_web_sm

# Clone the repository
git clone https://github.com/Lone-Wolfgang/DocuTrance.git

#Navigate to the docker folder and initiate an OpenSearch node
cd DocuTrance/search_engine/docker
docker compose up -d
```

## Setting Up OpenSearch with Docker

The provided `docker-compose.yml` installs **OpenSearch 2.13**, which includes semantic search capabilities. It also installs **OpenSearch Dashboards**, allowing you to browse and inspect your index via a web UI. Wait a few minutes for the containers to fully initialize.
If you're using the default settings, you can access OpenSearch Dashboards at:

http://localhost:5601/app/home#/


## Building the Index

```bash
#From the root directory, ~Docutrance/
cd search_engine
python ingest_pipeline.py
```


The Python script `ingest_pipeline.py` begins by taking a list of Wikipedia URLs and extracting their content for indexing. By default, it includes URLs for 180 country pages from *Wikipedia*.

Two dataframes are prepared for ingestion:

### Documents
Where each row represents a full page from Wikipedia. Entries include:

- **document_id**: Unique idientifier for each document
- **title**: The title of the Wikipedia page
- **title_embedding**: Vectorized representation of the title for semantic search.
- **body**: The text content of the Wikipedia page.
- **body_lemmatized**: Normalized representation of body text for keyword search.

### Segments
The Wikipedia corpus is split into overlapping segments of a few sentences that fit within the model context window. Entries include:

- **segment_id**: Unique identifier for each segment.
- **document_id**: The document from which the segment was extracted.
- **segment**: A few sentences from the Wikipedia page.
- **segment_embedding**: A vectorized represntation of the segment for semantic search.


### Launching the App

Once your index is built, browse it using `app.py`:

```bash
#From the root directory, ~Docutrance/
cd search_engine
streamlit run app.py
``` 

This rudimentary search engine provides a simple search bar interface. When a user enters a query, it ranks indexed Wikipedia pages based on relevance. The system performs both keyword and semantic searches, combining their results using Reciprocal Rank Fusion (RRF). Launch the app and try different queries to explore its capabilities and limitations. By default, it uses a multilingual model, allowing users to search in their preferred language.




