import os
import streamlit as st
import nest_asyncio
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
nest_asyncio.apply()

uploaded_file = st.file_uploader("Choose a file")

def load_data():
  docs = SimpleDirectoryReader(input_files = uploaded_file).load_data()
  #print(type(docs))
  docs[0]
  print(f"Loaded {len(docs)} docs")
  return docs
documents = load_data()

'''print(type(docs))
docs[0]
print(f"Loaded {len(docs)} docs")
'''
splitter = SentenceSplitter(chunk_size=2000,chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)
print(type(nodes))

Settings.llm = Groq(model = 'llama-3.3-70b-versatile' ,
    api_key="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")

Settings.embed_model = HuggingFaceEmbedding()

from llama_index.core import SummaryIndex
summary_index = SummaryIndex(nodes)

summary_query_engine = summary_index.as_query_engine(
    response_modes = "Tree Summarizer",
    use_async = True
)

from llama_index.core.tools import QueryEngineTool

summary_tool = QueryEngineTool.from_defaults(
    query_engine = summary_query_engine ,
    description= "Useful for summarization related to the given context"
)

response = summary_query_engine.query("Summarize the given document")
st.write(response)