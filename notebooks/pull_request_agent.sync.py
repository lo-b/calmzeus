# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
## Exploration of simple pull request (PR) agent
#
### Intro
# How to go from LLM response/answer to a commit (and PR)?
#
# - One way could be to create an agent. It could extract only relevant text in
# an answer and then use this to make the commit for a config task.
# - Another way could be to try playing around with the prompt to get a more
# exact/concise output.
# - Exploring possibility simple (A)ST-based indexing of source files.
# Ensures we keep track of the actual lines in source file of a snippet. Feed
# al this context forward into the LLM. Maybe it can use the additional (e.g.
# line number) context to be more precise in it's output.
#
# In this notebook, CST enrichment (for details see
# [here](./cst_indexing.sync.ipynb)) + prompt engineering is used.
#
### Goal
# Create an agent/chain to extract relevant code/text from LLM answer and
# create a commit.

# %% [markdown]
## Exploration
### Imports
# %%
import os

from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from qdrant_client import QdrantClient
from typing_extensions import Never

# %% [markdown]
### Define constants
# %%
QDRANT_COLLECTION_NAME = "simple-java-api"
VOYAGE_MODEL_NAME = "voyage-code-2"
MISTRAL_MODEL_NAME = "open-codestral-mamba"

# %%
assert load_dotenv(), ".env files exists and contains at least one variable"

# %% [markdown]
### Create (core) RAG flow
# Consists of:
# - indexing
# - retrieval
# - generation

# %% [markdown]
#### Indexing & retrieval
# Load documents and enrich their metadata with *context syntax trees* (CSTs).
# Add docs to vector store and create a retriever for the store.
# %%
embeddings = VoyageAIEmbeddings(model=VOYAGE_MODEL_NAME, batch_size=1)

client = QdrantClient(
    url=f"https://{os.environ['QDRANT_CLUSTER_ENDPOINT']}:6333",
    api_key=os.environ["QDRANT_API_KEY"],
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION_NAME,
    embedding=embeddings,
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 5, "lambda_mult": 0.25},
)

# %% [markdown]
#### Rephrase question
# Implement small chain to rephrase a user's question -- ideally to find a
# more similar document. Using [GPT-4o mini](
# https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/
# ) to rephrase question. As of writing, costs are as following:
# |in- or output|cost per million (1M) tokens|
# |---|---|
# |input|\$0.150|
# |output|\$0.600|

# %%
prompt: PromptTemplate = hub.pull("lo-b/rag-rephrase-assist-prompt")
llm = ChatOpenAI(model="gpt-4o-mini")

rephraser: RunnableSerializable[Never, str] = (
    {"question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
)

# %%
rephraser.invoke("Change app dev port to 7777")
