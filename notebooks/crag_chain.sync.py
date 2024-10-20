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
# Build a simple RAG chain to perform a config task on a simple Java
# application and evaluate its performance. Add tracing for rudimentary
# evaluation of the proposed change.

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
# - `MISTRAL_API_KEY`

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
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import Language
from langchain_voyageai import VoyageAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from rich import print as rprint

# %%
assert load_dotenv(), ".env file should be defined"

# %% [markdown]
### Define some consts

# %%
QDRANT_COLLECTION_NAME = "simple-java-api"
VOYAGE_MODEL_NAME = "voyage-code-2"
SRC_CODE_PATH = "/home/bram/projects/heavenlyhades/java/simple-api/"
MISTRAL_MODEL_NAME = "open-codestral-mamba"

# %% [markdown]
### Load source code in as documents

# %%
loader = GenericLoader.from_filesystem(
    SRC_CODE_PATH,
    glob="**/src/main/**/[!.]*",
    suffixes=[".java", ".properties"],
    parser=LanguageParser(Language.JAVA),
)
documents = loader.load()

# %%
print("loaded", len(documents), "files from disk")

# %% [markdown]
#### Sample document
# %%
rprint(documents[0])

# %% [markdown]
### Embed documents using Voyage.ai & store in vector DB
# %%
embeddings = VoyageAIEmbeddings(model=VOYAGE_MODEL_NAME, batch_size=1)

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
    collection_name=QDRANT_COLLECTION_NAME,
    vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
)

# %%
vector_store = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION_NAME,
    embedding=embeddings,
)

v_uuids = vector_store.add_documents(documents=documents, ids=uuids)

# %% [markdown]
#### Create vector db retriever
# %%
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 5, "lambda_mult": 0.25},
)

# %%
llm = ChatMistralAI(model_name=MISTRAL_MODEL_NAME)

# %%
retriever = vector_store.as_retriever()
prompt = hub.pull("lo-b/rag-config-assist-prompt")

# %% [markdown]
## Building the actual chain + tracing

# %%
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("Show me how to make my app run on port 7777")

# %% [markdown]
# ### LangSmith trace output
# Below, a print screen of the rag chain's execution trace where (rendered)
# input/output can be seen.
#
# The model gives the correct output (green) but also gives additional
# incorrect information (red). It is *not necessary* to 'change' the main
# class. The change is even -- I think -- what's already in the class too...
# ![Voyage-Mistral-RAG](../assets/langsmith-trace-simple-java-api-voyage-mistral.png)

# NOTE: In previous runs model output contained additional, correct, info --
# something like: "this is how to run your app". Additional information could
# be useful but can also pollute output/complicate PR creation.

## Conclusion
# Using a simple prompt
# (see 👉 [here](https://smith.langchain.com/hub/lo-b/rag-config-assist-prompt))
# to guide the generation model worked pretty decent.

# Initially, a local llama model was used which contained the correct answer,
# but it contained some quirks, highlighted in red:
# ![Voyage-Llama-RAG](../assets/langsmith-trace-simple-java-api-voyage-local-llama.png)

# Swapping the local Llama model with Codestral showed better results. A
# downside is that it can't easily be hosted locally. I need a *whole stick* of
# 16GB extra RAM to run it on my PC... After which, it most definitely won't
# fit into GPU memory, so I do not know what the latency will be.

## Improvements
# - Prompt engineering: ensure model only outputs desired *commitable* change
# -- as of now, parsing output can be done but isn't ideal.
# - Track code structure: keep track which lines of a file are fed to the model
# as docs. Together with better prompt engineering above, model might be nudged
# into the direction of generating a '*pure*' PR change.
