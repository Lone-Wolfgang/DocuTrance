# app.py
import streamlit as st
from docutrance.ux import (
    setup_page,
    initialize_session_state,
    render_main_input
)
from docutrance.search import hybrid_search_pipeline, preprocess_input, compose_query_body
from configs.app import (
    CLIENT,
    ENCODER,
    LEMMATIZER,
    TAB_TITLE,
    USER_PROMPT,
    DOCUMENTS,
    JOBS
)
import pandas as pd

# --- Setup page
setup_page(TAB_TITLE)

# Initialize session state
if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""
if "reset_triggered" not in st.session_state:
    st.session_state["reset_triggered"] = False

# Reset logic
if st.session_state.reset_triggered:
    st.session_state.query_input = ""
    st.session_state.reset_triggered = False
    st.rerun()

# Sidebar reset button (sets a flag)
if st.sidebar.button("🔄 Reset", type="secondary"):
    st.session_state.reset_triggered = True

# Main input
st.text_input(USER_PROMPT[0], key="query_input")

# Process query
processed_input = preprocess_input(st.session_state.query_input, LEMMATIZER, ENCODER)


# --- Dynamically incorporate user input into queries
results = hybrid_search_pipeline(
    DOCUMENTS,
    processed_input,
    CLIENT,
    JOBS
)

if results.empty:
    st.warning('No Results found.')

for _, row in results.iterrows():
    title = f"### [{row['title']}]({row['url']})"
    st.markdown(title)

    if row.get('highlights'):
        highlights = [h for h in row['highlights'] if h]
        highlights = '---\n\n' + '\n\n---\n\n'.join(highlights)
        st.markdown(highlights)
    
    st.markdown('---')