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
from typing import Any, Optional, TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language.language_parser import (
    LanguageParser,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import (
    Language as SplitterLanguage,
)
from langchain_voyageai import VoyageAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rich import print as rprint
from tree_sitter import Language, Node, Parser, Tree
from typing_extensions import Never

# %% [markdown]
### Define constants
# %%
QDRANT_COLLECTION_NAME = "cst-enriched-simple-java-api"
VOYAGE_MODEL_NAME = "voyage-code-2"
MISTRAL_MODEL_NAME = "open-codestral-mamba"

# %%
assert load_dotenv(), ".env files exists and contains at least one variable"

# %% [markdown]
### Create full RAG flow
# Consists of:
# - indexing
# - retrieval
# - generation

# %% [markdown]
#### Indexing
# Load documents and enrich their metadata with *context syntax trees* (CSTs).
# Add docs to vector store and create a retriever for the store.


# %% [markdown]
##### Create java and properties parser

# %%
JPROP_LANGUAGE = Language("../parsers/ts-properties.so", "properties")
JAVA_LANGUAGE = Language("../parsers/ts-java.so", "java")

# create parsers
java_parser = Parser()
java_parser.set_language(JAVA_LANGUAGE)

jprop_parser = Parser()
jprop_parser.set_language(JPROP_LANGUAGE)

# %% [markdown]
##### Code to convert tree to dictionary
# Create *pre-order traversal* (first process a node itself, then its children)
# algorithm to convert CST to JSON.


# %%
class NodeDict(TypedDict):
    grammar_name: str
    text: str
    start: tuple[int, int]
    end: tuple[int, int]
    children: list[Any]


def node_to_dict(node: Node) -> NodeDict:
    node_dict: NodeDict = {
        "grammar_name": node.grammar_name,
        "text": bytes.decode(node.text),
        "start": node.start_point,
        "end": node.end_point,
        "children": [],
    }

    for child in node.children:
        node_dict["children"].append(node_to_dict(child))

    return node_dict


# %% [markdown]
##### Load files

# %%
java_code_dir = "/home/bram/projects/heavenlyhades/java/simple-api/"
loader = GenericLoader.from_filesystem(
    java_code_dir,
    glob="**/src/main/**/[!.]*",
    suffixes=[".java", ".properties"],
    parser=LanguageParser(SplitterLanguage.JAVA),
)
documents = loader.load()
print("loaded", len(documents), "docs")


# %% [markdown]
##### Enrich metadata with CST
# %%
def construct_cst(doc: Document) -> Optional[NodeDict]:
    doc_source = doc.metadata["source"]
    cst: Optional[NodeDict] = None
    if ".properties" in doc_source:
        cst = jprop_parser.parse(str.encode(doc.page_content))
    if ".java" in doc_source:
        cst = java_parser.parse(str.encode(doc.page_content))

    return node_to_dict(cst.root_node)


for doc in documents:
    assert (
        ".properties" or ".java" in doc.metadata["source"]
    ), "only set up parsers for java/properties files"

    doc.metadata["cst"] = construct_cst(doc)

# %% [markdown]
##### Add documents to collection

# %%
embeddings = VoyageAIEmbeddings(model=VOYAGE_MODEL_NAME, batch_size=1)

# %%
sample_text = "69-420"  # example text to determine embedding size
embedding_size = len(embeddings.embed_query(sample_text))


client = QdrantClient(
    url=f"https://{os.environ['QDRANT_CLUSTER_ENDPOINT']}:6333",
    api_key=os.environ["QDRANT_API_KEY"],
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION_NAME,
    embedding=embeddings,
)

if not client.collection_exists(QDRANT_COLLECTION_NAME):
    _ = client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
    )
    uuids = [str(uuid4()) for _ in range(len(documents))]
    v_uuids = vector_store.add_documents(documents=documents, ids=uuids)


# %% [markdown]
##### Construct retriever for vector DB
# %%
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 5, "lambda_mult": 0.25},
)

# %% [markdown]
#### Retrieval
# Implement small chain to rephrase a user's question -- ideally to find a
# more similar document. Using [GPT-4o mini](
# https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/
# ) to rephrase question. As of writing, costs are as following:
# |in- or output|cost per million (1M) tokens|
# |---|---|
# |input|\$0.150|
# |output|\$0.600|

# %%
rephrase_prompt: PromptTemplate = hub.pull("lo-b/rag-rephrase-assist-prompt")
gpt_4o_mini = ChatOpenAI(model="gpt-4o-mini")

rephrased_retriever: RunnableSerializable[Never, list[Document]] = (
    {"question": RunnablePassthrough()}
    | rephrase_prompt
    | gpt_4o_mini
    | StrOutputParser()
    | retriever
)

# %% [markdown]
#### Generation (full chain)
# %%
mistral = ChatMistralAI(model_name=MISTRAL_MODEL_NAME)
config_prompt: PromptTemplate = hub.pull("lo-b/rag-config-assist-prompt")
generate: RunnableSerializable[Never, str] = (
    {"context": rephrased_retriever, "question": RunnablePassthrough()}
    | config_prompt
    | mistral
    | StrOutputParser()
)

# %%
answer = generate.invoke("Change app dev port to 7777")

# %%
rprint(answer)
