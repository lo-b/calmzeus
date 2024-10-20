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
# together *what exact* `sed` (what else can I use to do a CLI-based text edit?)
# command to run.
#
# Why not use an AI (agent -- another LLM) to do this for us 😄.
#
# Given documents are parsed using
# [treesitter](https://tree-sitter.github.io/tree-sitter/)
# for additional metadata in the form of an AST, how accurate is an answer
# when this additional metadata is fed in? For a very simple task it does well:
# ![Treesitter infused prompt](../assets/treesitter-context-prompt-gpt4-example.png)

# Again, there's additional bloat in the response but running the exact command
# against the file gives back the desired commitable change:
# ![git diff after cmd](../assets/diff-after-sed-command.png)

### Goal
# Explore treesitter ASTs and add parsed tree output as metadata to a loaded
# document.
# > 🤔 Note that in previous exploration, chunk size was set to low. Therefore,
# a single file probably got split into multiples documents containing a
# file/code snippet where a snippet can contain something like
# `//Code for @SpringBootApp...` to refer to another related snippet/doc --
# keep this in mind!
