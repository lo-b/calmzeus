# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Building a simple LCEL chain with tracing
# ## Intro
# How easy is it to create a simple RAG chain -- and how feasible is it to
# create an LLM-powered *'config-changer'*?
#
# A tool that is able to make configuration changes in a project given a user's
# instructions. E.g.:
# #### 🧑:
# "_Show me how to make my app run on port 7777_"

# ---
# #### 🤖:
# It looks like your project is a **Java-based application powered by Spring Boot**. In order to change the port the application is running on you have to: ...
#
# I've **created a PR with the necessary changes** you to review:
# \<link>

# ## Goal
# Build a chain with tracing to see how a rudimentary vector store and local
# llama model perform for a simple configuration task.

## Before you start
# [voyage-code-2](https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/)
# is used as the embedding model. [qdrant](https://qdrant.tech/) is used to store embeddings.

# Create an `.env` file and set the following env variables:
# - `VOYAGE_API_KEY` (api key)
# - `LANGCHAIN_TRACING_V2` (`true` | `false`)
# - `LANGCHAIN_ENDPOINT`
# - `LANGCHAIN_API_KEY`
# - `LANGCHAIN_PROJECT`
# - `QDRANT_API_KEY`
# - `QDRANT_CLUSTER_ENDPOINT`

# Grab a copy of the Java source code, available
# [here](https://github.com/lo-b/heavenlyhades/tree/main/java/simple-api).
# The project inside the repo contains code of a (completed) Spring Boot
# tutorial of how to create an API.

# %% [markdown]
### Imports
# %%
import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language.language_parser import (
    LanguageParser,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import Language
from langchain_voyageai import VoyageAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from rich import print as rprint

# %%
assert load_dotenv(), ".env file should be defined"

# %% [markdown]
### Load source code in as documents
# %%
src_code_dir = "/home/bram/projects/heavenlyhades/java/simple-api/"
loader = GenericLoader.from_filesystem(
    src_code_dir,
    glob="**/src/main/**/[!.]*",
    suffixes=[".java", ".properties"],
    parser=LanguageParser(Language.JAVA),
)
documents = loader.load()

# %%
print("loaded ", len(documents), " from disk")

# %% [markdown]
#### Sample document
# %%
rprint(documents[0])

# %% [markdown]
### Embed documents using Voyage.ai & store in vector DB
# %%
embeddings = VoyageAIEmbeddings(model="voyage-code-2", batch_size=1)

# %%
sample_text = "69-420"  # example text to determine embedding size
embedding_size = len(embeddings.embed_query(sample_text))

uuids = [str(uuid4()) for _ in range(len(documents))]

# %%
client = QdrantClient(
    url=f"https://{os.environ['QDRANT_CLUSTER_ENDPOINT']}:6333",
    api_key=os.environ["QDRANT_API_KEY"],
)

_ = client.create_collection(
    collection_name="simple-java-api",
    vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
)

# %%
vector_store = QdrantVectorStore(
    client=client,
    collection_name="simple-java-api",
    embedding=embeddings,
)

# %%
v_uuids = vector_store.add_documents(documents=documents, ids=uuids)

# %% [markdown]
#### Create vector db retriever
# %%
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 5, "lambda_mult": 0.25},
)

# %%
# TODO: Ollama setup
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0,
    num_gpu=1,
)

# %%
retriever = vector_store.as_retriever()
# TODO: make prompt public or refactor
prompt = hub.pull("simplig-crag-config-prompt")

# %% [markdown]
# # Building the actual chain + tracing

# %%
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("Show me how to make my app run on port 7777")
