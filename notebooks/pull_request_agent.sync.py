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
from langchain_mistralai.chat_models import ChatMistralAI
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
### Recreate simple RAG chain
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

prompt: PromptTemplate = hub.pull("lo-b/rag-config-assist-prompt")

llm = ChatMistralAI(model_name=MISTRAL_MODEL_NAME)

rag_chain: RunnableSerializable[Never, str] = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# %%
rag_config_answer: str = rag_chain.invoke("Show me how to make my app run on port 7777")

# %%
rag_config_answer
