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
# # Exploration of simple pull request (PR) agent
#
# ## Intro
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
# ## Goal
# Create two agents:
# 1. one agent which will change a file
# 2. another agent to commit the change using git and create a pull request,
# returning its link after creation.

# %% [markdown]
# # Exploration
# ## Imports
# %%
import os
import subprocess
from typing import Any, Literal, Optional, TypedDict
from uuid import uuid4

from dotenv import load_dotenv
from IPython.display import Image
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
from langchain_core.tools import tool
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import (
    Language as SplitterLanguage,
)
from langchain_voyageai import VoyageAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rich import print as rprint
from tree_sitter import Language, Node, Parser
from typing_extensions import Never

# %% [markdown]
# ## Define constants
# %%
QDRANT_COLLECTION_NAME = "cst-enriched-simple-java-api"
VOYAGE_MODEL_NAME = "voyage-code-2"
MISTRAL_MODEL_NAME = "open-codestral-mamba"

# %%
assert load_dotenv(), ".env files exists and contains at least one variable"

# %% [markdown]
# ## Create full RAG flow
# Consists of:
# - indexing
# - retrieval
# - generation

# %% [markdown]
# ### Indexing
# Load documents and enrich their metadata with *context syntax trees* (CSTs).
# Add docs to vector store and create a retriever for the store.


# %% [markdown]
# #### Create java and properties parser

# %%
JPROP_LANGUAGE = Language("../parsers/ts-properties.so", "properties")
JAVA_LANGUAGE = Language("../parsers/ts-java.so", "java")

# create parsers
java_parser = Parser()
java_parser.set_language(JAVA_LANGUAGE)

jprop_parser = Parser()
jprop_parser.set_language(JPROP_LANGUAGE)

# %% [markdown]
# #### Code to convert tree to dictionary
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
# #### Load files

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
# #### Enrich metadata with CST
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
# #### Add documents to collection

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
# #### Construct retriever for vector DB
# %%
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 5, "lambda_mult": 0.25},
)

# %% [markdown]
# ### Retrieval
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
# ### Add tool for calling `sed` CLI command
# Define tool for running `sed` CLI command with the given 'cmd_args'.
# %%
@tool
def run_sed_cmd(cmd_args: list[str]) -> str:
    """
    Use Streaming Editor CLI (sed) command and arguments to manipulate text.
    """
    # WARNING: potential security & system risk if allowed to call ANY task;
    cmd = ["sed"] + cmd_args

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        # Log the error message and raise the exception
        error_message = (
            "Command failed "
            f"with return code {e.returncode}. Error: {e.stderr.strip()}"
        )
        raise RuntimeError(error_message) from e


# %% [markdown]
# ### Generation (full chain)
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


# %% [markdown]
# ### Create agent using LangGraph
# Use LangGraph to create an agent that calls the 'sed' tool -- think of chains as graphs, where some state gets passed and
# is updated, throughout the chain.
# %%
tools = [run_sed_cmd]

tool_node = ToolNode(tools)

model = gpt_4o_mini.bind_tools(tools)


# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["sed_tool", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "sed_tool" node
    if last_message.tool_calls:
        return "sed_tool"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("gpt4o-mini", call_model)
workflow.add_node("sed_tool", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "gpt4o-mini")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "gpt4o-mini",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("sed_tool", "gpt4o-mini")

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)

# %% [markdown]
# ### Visualize graph
# %%
display(Image(app.get_graph(xray=True).draw_mermaid_png()))

# %% [markdown]
# ### Test agent with `sed` tool
# #### Write line to `test_file.txt`
# %%
%%bash
echo "serendipity" > test_file.txt


# %% [markdown]
# #### Ask agent to change manipulate text
# %%
config = {"configurable": {"thread_id": "1"}}
file_location = "/home/bram/projects/calmzeus/notebooks/test_file.txt"
user_input = f"Given the file at the location `{file_location}` change the text 'serendipity' to Serendipitous"
events = app.stream({"messages": [("user", user_input)]}, config, stream_mode="values")
for event in events:
    event["messages"][-1].pretty_print()

# %% [markdown]
# #### check file
# %%
with open(file_location, "r") as s:
    for l in s.readlines():
        print(l)

# %% [markdown]
# #### remove file
# %%
%%bash
rm /home/bram/projects/calmzeus/notebooks/test_file.txt
