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
# Create a vector store of a very rudimentary Java (spring-boot) API. Check how
# well retrieval of relevant files works. I.e. given the prompt:

# *"This file is used to configure the application's port."*

# Ideally the store should give back documents that are similar to the prompt,
# e.g.: `application.properties`.

## Before you start
# Create an `.env` file and add & set the `VOYAGE_API_KEY` key.

# [voyage-code-2](https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/)
# is used as the embedding model. After searching for 'best code embedding
# models' this came up, so let's give it a try!

# Grab a copy of the Java source code, available
# [here](https://github.com/lo-b/heavenlyhades/tree/main/java/simple-api)


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
### Loading documents (`.java`/`.properties` files)
# Load source code in as documents, only grabbing 'properties' and Java files.

# %% document loading
csharp_code_dir = "/home/bram/projects/heavenlyhades/java/simple-api/"
loader = GenericLoader.from_filesystem(
    csharp_code_dir,
    glob="**/src/main/**/[!.]*",
    suffixes=[".java", ".properties"],
    parser=LanguageParser(Language.JAVA),
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
    search_kwargs={"k": 4, "fetch_k": 5, "lambda_mult": 0.25},
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
# Not (what I would say) the most similar doc, which would be: `application.properties`.

# %%
config_query_similarity[1].metadata["source"]

# %% [markdown]
# It however seems to finds the file I expected as 'second-most-similar'
# document.

### Asking about API configuration
# %%
endpoint_query_similarity = retriever.invoke(
    "The name of the endpoint is configured here."
)

print(endpoint_query_similarity[0].id)
print(endpoint_query_similarity[0].metadata["source"])
print(endpoint_query_similarity[0].metadata["language"])
print(endpoint_query_similarity[0].page_content)

# %% [markdown]

# This is actually the right place! 🥳

# However, if we ask it in a similar way:
# %%
resource_query_similarity = retriever.invoke(
    "Name of an API resource is configured here."
)

print(resource_query_similarity[0].id)
print(resource_query_similarity[0].metadata["source"])
print(resource_query_similarity[0].metadata["language"])
print(resource_query_similarity[0].page_content)

# %% [markdown]
# Would expect the previous answer (controller class) again.

# %% [markdown]
## Conclusion

# Using voyage embedding + Java seems to give better results (for this simple
# setup), compared to C# + voyage vector store.

# Fair, since embedding model was trained on Java (see [Quantitative
# Evaluation](https://blog.voyageai.com/2024/01/23/voyage-code-2-elevate-your-code-retrieval/)).
#
# Assumed it would handle arbitrary programming languages better, especially
# since I'm feeding it _'Microsoft Java'_ (C#) instead of Java.
