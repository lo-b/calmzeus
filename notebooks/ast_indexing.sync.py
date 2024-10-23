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
## Explore AST-based indexing
### Intro
# It's not trivial to *post process* LLM output and then manually piece
# together *what exact* `sed` (or similar tool to edit text) command to run.
#
# A possible solution could be to hook up another model to reason about
# the action to take (also called an *agent*) to do this for us 😄. However,
# let's take a step back and see if feeding additional context works.
#
# Given documents are parsed using
# [treesitter](https://tree-sitter.github.io/tree-sitter/)
# for additional metadata in the form of a context syntax tree (CST), how
# accurate is an answer when this additional metadata is fed in?
# For a very simple task it does well:
# ![Treesitter infused prompt](../assets/treesitter-context-prompt-gpt4-example.png)

# Again, there's additional bloat in the response but running the exact command
# against the file gives back the desired commitable change:
# ![git diff after cmd](../assets/diff-after-sed-command.png)

### Goal
# Explore treesitter CSTs and add parsed tree output as metadata to a loaded
# document.
# > 🤔 Note that in previous exploration, chunk size was set to low. Therefore,
# a single file probably got split into multiples documents containing a
# file/code snippet where a snippet can contain something like
# `//Code for @SpringBootApp...` to refer to another related snippet/doc --
# keep this in mind!

# %% [markdown]
## Exploration
### Imports
# %%
import json
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language.language_parser import (
    LanguageParser,
)
from langchain_core.documents import Document
from langchain_text_splitters import (
    Language as SplitterLanguage,
)
from rich import print as rprint
from tree_sitter import Language, Node, Parser, Tree

# %% [markdown]
### load env vars
# %%
assert load_dotenv(), ".env file with variable present"

# %% [markdown]
### Load & create parser
# %%
# load in parser libs for 'properties' file
LANG_NAME = "properties"
JPROP_LANGUAGE = Language("../parsers/ts-properties.so", LANG_NAME)

# create parser
parser = Parser()
parser.set_language(JPROP_LANGUAGE)

# %% [markdown]
### Parse file and create CST (context syntax tree)
# %%
# mock `application.properties` file
file_contents: bytes = bytes(
    "debug=true\nanotherPropKey='propValue'",
    "utf-8",
)
tree: Tree = parser.parse(file_contents)

# %% [markdown]
#### Some info about the tree
##### Root node
# %%
rprint(tree.root_node)
# %% [markdown]
# A node (for the file) spanning from line 0, column 0 to line 1 column 26.

# %% [markdown]
##### tree as a 'string' expression
# %%
rprint(tree.root_node.sexp())

# %% [markdown]
##### descendants
# %%
print(tree.root_node.descendant_count)


# %% [markdown]
### Convert tree to dictionary
# Create *pre-order traversal* (first process a node itself, then its children)
# algorithm to convert CST to JSON.


# %%
class NodeDict(TypedDict):
    grammar_name: str
    text: str
    start: tuple[int, int]
    end: tuple[int, int]
    children: list[Any]  # Ideally of type NodeDict... 🔁


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
### pretty print JSON of parsed CST
# %%
rprint(json.dumps(node_to_dict(tree.root_node), indent=2))

# %% [markdown]
### Add CST to document metadata
# Add *context syntax tree* (CST) dictionary to `.properties` files because
# only this parser (language) is set up.
#### Load docs

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
#### Enrich metadata with CST
# %%
def construct_cst(doc: Document):
    cst = parser.parse(str.encode(doc.page_content))
    return node_to_dict(cst.root_node)


for doc in documents:
    if ".properties" in doc.metadata["source"]:
        doc.metadata["cst"] = construct_cst(doc)

enriched_prop_doc = [
    doc for doc in documents if ".properties" in doc.metadata["source"]
][0]

rprint(enriched_prop_doc)

# %% [markdown]
## Conclusion
# Using treesitter to parse Java `.properties` files to get a context tree,
# proved pretty straight forward. Note, that for a file *containing a single property*
# the CST is already pretty big.

### Improvements
# Prevent duplication of file content ('debug=false'); currently duplicated
# in file, property and key + value texts. Maybe it's enough to only keep key
# and values?
