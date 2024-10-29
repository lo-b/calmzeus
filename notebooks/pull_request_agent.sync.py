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
from typing import Any, Literal, Optional, TypedDict, Annotated
from uuid import uuid4

from dotenv import load_dotenv
from IPython.display import Image
from langchain import hub
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language.language_parser import (
    LanguageParser,
)
from operator import add
from collections.abc import Sequence
from langgraph.graph.message import add_messages
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableSerializable,
    RunnablePick
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
# Use LangGraph to create an agent that calls the 'sed' tool -- think of 
# chains as graphs, where some state gets passed and is updated, throughout 
# the chain.
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
# #### Visualize graph
# %%
display(Image(app.get_graph(xray=True).draw_mermaid_png()))

# %% [markdown]
# ### Test agent with `sed` tool
# #### Write line to `test_file.txt`
# %%
%%bash
echo "serendipity" > test_file.txt


# %% [markdown]
# #### Ask agent to manipulate text
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

# %% [markdown]
# ### Manual agent call flow
# Manually call agents and their tools to change, commit and create a PR.

# #### list current git changes
# %%
%%bash
git -C /home/bram/projects/git_test_repo status

# %% [markdown]
# #### use sed to create change
# %%
file_to_manipulate = "/home/bram/projects/git_test_repo/some_file.txt"
manipulate_prompt = (
    "Add a line saying 'added line' "
    f"to the file located at `{file_to_manipulate}`"
)
events = app.stream({"messages": [("user", manipulate_prompt)]}, config, stream_mode="values")
for event in events:
    event["messages"][-1].pretty_print()


# %% [markdown]
# #### show diff after agent tool call
# %%
%%bash
git -C /home/bram/projects/git_test_repo diff

# %% [markdown]
# #### define git commit tool and add it to the tools
# %%
@tool
def git_tool(sub_cmd: str, cmd_args: list[str]):
    """
    Use git sub command (sub_cmd) to create a commit (commit) or stage changes.
    """
    # WARNING: potential security & system risk if allowed to call ANY task;
    cmd = ["git", sub_cmd] + cmd_args

    try:
        print("DEBUG:", "try running following command:\n", f"==> {cmd} <==")
        result = subprocess.run(
            cmd,
            cwd="/home/bram/projects/git_test_repo/",
            check=True,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            capture_output=True,
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
# ### Create agent that has git as a tool
# Use LangGraph to create an agent that calls git tooling
# %%
tools = [git_tool]
tool_node = ToolNode(tools)
model = gpt_4o_mini.bind_tools(tools)

def should_continue(state: MessagesState) -> Literal["git_tool", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "git_tool"
    return END

workflow = StateGraph(MessagesState)
workflow.add_node("gpt4o-mini", call_model)
workflow.add_node("git_tool", tool_node)
workflow.add_edge(START, "gpt4o-mini")
workflow.add_conditional_edges("gpt4o-mini", should_continue)
workflow.add_edge("git_tool", "gpt4o-mini")
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# %% [markdown]
# #### Visualize graph
# %%
display(Image(app.get_graph(xray=True).draw_mermaid_png()))

# %% [markdown]
# #### create new branch
# %%
branch_prompt = (
        f"Create a new branch named 'bot/config-change' and change to it"
)
events = app.stream({"messages": [("user", branch_prompt)]}, config, stream_mode="values")
for event in events:
    event["messages"][-1].pretty_print()

# %% [markdown]
# #### add/stage changes
# %%
file_to_stage = "/home/bram/projects/git_test_repo/some_file.txt"
stage_prompt = (
        f"Add changes to be staged in the file ({file_to_stage})"
)
events = app.stream({"messages": [("user", stage_prompt)]}, config, stream_mode="values")
for event in events:
    event["messages"][-1].pretty_print()

# %% [markdown]
# #### commit changes
# %%
commit_prompt = (
        f"create a commit, prefix the title with 'bot:' "
        "to indicate a non human wrote the commit"
)
events = app.stream({"messages": [("user", commit_prompt)]}, config, stream_mode="values")
for event in events:
    event["messages"][-1].pretty_print()

# %% [markdown]
# #### push new branch + changes
# %%
push_changes_prompt= (
        f"Push the new changes"
)
events = app.stream({"messages": [("user", push_changes_prompt)]}, config, stream_mode="values")
for event in events:
    event["messages"][-1].pretty_print()


# %% [markdown]
# #### add tool for creation of a GitHub PR
# %%
@tool
def gh_pr_create(title: str, description: str):
    """
    Use GitHub CLI command to create a Pull Request.
    """
    # WARNING: potential security & system risk if allowed to call ANY task;
    cmd = (
        f"gh pr create --title '{title}' --body '{description}'"
           )

    try:
        result = subprocess.run(
            cmd,
            cwd="/home/bram/projects/git_test_repo",
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
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

# %# %% [markdown]
# ### Create agent that has PR creation tool
# Use LangGraph to create an agent that calls GitHub cli to create a PR
# %%
tools = [gh_pr_create]

tool_node = ToolNode(tools)

model = gpt_4o_mini.bind_tools(tools)


def should_continue(state: MessagesState) -> Literal["gh_pr_create", END]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "gh_pr_create"
    return END

workflow = StateGraph(MessagesState)
workflow.add_node("gpt4o-mini", call_model)
workflow.add_node("gh_pr_create", tool_node)
workflow.add_edge(START, "gpt4o-mini")
workflow.add_conditional_edges("gpt4o-mini", should_continue)
workflow.add_edge("gh_pr_create", "gpt4o-mini")
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# %% [markdown]
# #### Visualize graph
# %%
display(Image(app.get_graph(xray=True).draw_mermaid_png()))

# %% [markdown]
# #### create PR and return a link
# %%
repo_root = "/home/bram/projects/git_test_repo"
create_pr_prompt = (
    f"In the following git repository root '{repo_root}' "
    "create a GitHub pull request. "
    "Ensure it is clear you 'PRagent' created it. "
    "Only return the link to the PR you created."
)
events = app.stream({"messages": [("user", create_pr_prompt)]}, config, stream_mode="values")
for event in events:
    event["messages"][-1].pretty_print()


# %% [markdow]
# Putting it all together
# %%
class AgentState(MessagesState):
    docs: Annotated[list[Document], add]
    user_question: str
    rephrased_question: str


# %%
def rephrased_retrieval(state: AgentState):
    print("---REPHRASE---")
    messages = state["messages"]
    # NOTE: dirty hack to get the first line containing question (could be cleaner)
    question: str = messages[0].content.splitlines()[0]
    rephrase_prompt: PromptTemplate = hub.pull("lo-b/rag-rephrase-assist-prompt")
    rephrase_chain = (
        {"question": RunnablePassthrough()}
        | rephrase_prompt
        | gpt_4o_mini
        | StrOutputParser()
    )

    rephrased_question: str = rephrase_chain.invoke(question)

    docs = retriever.invoke(rephrased_question)

    print(len(docs), "documents retrieved")

    return {
        "messages": [rephrased_question],
        "docs": docs,
        "user_question": question,
        "rephrased_question": rephrased_question,
    }


def generate(state: AgentState):
    print("---GENERATE---")
    question: str = state["user_question"]

    docs: list[Document] = state["docs"]

    mistral = ChatMistralAI(model_name=MISTRAL_MODEL_NAME)
    config_prompt: PromptTemplate = hub.pull("lo-b/rag-config-assist-prompt")

    generate: RunnableSerializable[dict, str] = (
        {
            "context": RunnablePick(keys=["context"]),
            "question": RunnablePick(keys=["question"])
        }
        | config_prompt
        | mistral
        | StrOutputParser()
    )

    response = generate.invoke({"context": docs, "question": question})
    return {"messages": [response]}


# %% [markdown]
# #### Create workflow
# %%
config_rag_flow = StateGraph(MessagesState)
config_rag_flow.add_node("rephrased-retrieval", rephrased_retrieval)
config_rag_flow.add_node("rag", generate)
config_rag_flow.add_edge(START, "rephrased-retrieval")
config_rag_flow.add_edge("rephrased-retrieval", "rag")
config_rag_flow.add_edge("rag", END)
checkpointer = MemorySaver()
config_rag_app = config_rag_flow.compile(checkpointer=checkpointer)

# %% [markdown]
# #### Visualize flow
# %%
display(Image(config_rag_app.get_graph(xray=True).draw_mermaid_png()))

# %% [markdown]
# ### test spin 🙏
# %%
config_change_prompt = "Ensure debugging is turned off"
events = config_rag_app.stream({"messages": [("user", config_change_prompt)]}, config, stream_mode="values")
for event in events:
    event["messages"][-1].pretty_print()

