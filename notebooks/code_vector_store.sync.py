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
# # Exploration of code vector store
## Intro
# Extracting context for an LLM config assistant to use could be realised by
# just loading in all project/repository's source files that contain code or
# configuration.

## Goal
# Create a vector store of a very rudimentary C# (net8) API. Check how well
# retrieval of relevant files works. I.e. given the prompt:

# *"This file is used to configure the application's port."*

# Ideally the store should give back documents that are similar to the prompt,
# e.g.: `appsettings.json`.

## Before you start
# Create an `.env` file and add & set the `VOYAGE_API_KEY` key.

# [voyage-code-2](https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/)
# is used as the embedding model. After searching for 'best code embedding
# models' this came up, so let's give it a try!

# Grab a copy of the C# source code, available
# [here](https://github.com/lo-b/heavenlyhades/tree/main/csharp/simple-api)


## Let's get to it
# %%
from uuid import uuid4

import faiss
from dotenv import load_dotenv
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language.language_parser import (
    LanguageParser,
)
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import Language
from langchain_voyageai import VoyageAIEmbeddings
from rich import print as rprint

# %%
assert load_dotenv(), "API vars should be defined in .env file"

# %% [markdown]
### Loading documents (`.cs`/`.json` files)
# Load source code in as documents, only grabbing JSON and C# files.

# %% document loading
csharp_code_dir = "/home/bram/projects/heavenlyhades/csharp/simple-api/SimpleApi"
loader = GenericLoader.from_filesystem(
    csharp_code_dir,
    glob="**/[!.]*",
    # NOTE: exclude below doesn't seem to work recursively :(
    exclude=[
        f"{csharp_code_dir}/obj/**",
        f"{csharp_code_dir}/**/Debug/net8.0/**",
    ],
    suffixes=[".cs", ".json"],
    parser=LanguageParser(Language.CSHARP, parser_threshold=10),
)
documents = loader.load()

# %% [markdown]
### Print out the loaded documents

# %%
print("Total docs: ", len(documents))

print("Doc (meta)data")
for doc in documents:
    print("METADATA:")
    for k, v in doc.metadata.items():
        print(f"=>{k}:\n\t{v}")
    print("JSON DUMP:")
    print(doc.model_dump_json())
    print("=" * 70)

# %% [markdown]
### Check document content loaded

# %%
for doc in documents:
    print("FILE: ", doc.metadata["source"])
    print("CONTENT: ")
    print(doc.page_content)
    print("=" * 70)

# %% [markdown]
### Initialize embedding model

# %%
embeddings = VoyageAIEmbeddings(model="voyage-code-2", batch_size=1)

# %% [markdown]
### Create vector store
# Initialize simple vector store using [FAISS](https://faiss.ai/index.html).
# Use voyage model initialized above as the embedding function.

# %%
sample_text = "69-420"  # example text to determine embedding size
embedding_size = len(embeddings.embed_query(sample_text))
index = faiss.IndexFlatL2(embedding_size)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

uuids = [str(uuid4()) for _ in range(len(documents))]

# %%
v_uuids = vector_store.add_documents(documents=documents, ids=uuids)

# %% [markdown]
### Similarity search | moment of truth 🙏
# %%
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1, "fetch_k": 5, "lambda_mult": 0.25},
)

config_query_similarity = retriever.invoke(
    "This file is used to configure the application's port."
)

# %%
print(config_query_similarity[0].id)
print(config_query_similarity[0].metadata["source"])
print(config_query_similarity[0].metadata["language"])
print(config_query_similarity[0].page_content)

# %% [markdown]
# Great! The most similar document returned is (what I would say) the most
# relevant file.


# %% [markdown]
### If only it were this easy 🥹
# %%
di_query_similarity = retriever.invoke(
    "This file is used to setup .NET 8 dependency injection"
)

print(di_query_similarity[0].id)
print(di_query_similarity[0].metadata["source"])
print(di_query_similarity[0].metadata["language"])
print(di_query_similarity[0].page_content)

# %% [markdown]
# Again the same `appsettings.json` file is returned ... I would expect the
# `Program.cs` file to be returned.

# %% [markdown]
### And now ... For something completely 'different'
# Thought I'd query the store for something completely unrelated (about weather)
# to see if it would also return the same file. Yet it _does_ seem to create
# relevant embeddings, capturing some context.
# %%
results = vector_store.similarity_search_with_score("Will it be hot tomorrow?", k=1)
for res, score in results:
    rprint(f"[SIM={score:3f}]\n")
    print(f"{'='*70}\n")
    rprint(res.page_content)
    print(f"\n{'='*70}")
    rprint(res.metadata)

# %% [markdown]
# The most similar doc returned is a weather-related record (C# keyword).

# Apparently, scaffolding a new .NET 8 API project using `dotnet` cli, creates
# a simple API that returns a weather 'forecast'.

# %% [markdown]
## Conclusion
# Setup of a vector store is fairly easy. Quality of embeddings is subpar.
# Similarity search for both queries (about 'dependency injection' and 'port
# configuration') return the same document: `appsettings.json`.

### Potential improvements:
# - pre-process documents
# - deep dive on [Existing Approaches to Code Embedding](https://www.unite.ai/code-embedding-a-comprehensive-guide/):
# token, tree or graph-based.
# - use 'better' code embedding model
# - prompt engineering for store retrieval
# - reranking
# - add grader (see LangChain's *corrective* RAG example)
