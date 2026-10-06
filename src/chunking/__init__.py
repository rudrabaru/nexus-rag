"""
Chunking: convert documents into retrieval units that follow heading structure.

Submodules are imported explicitly by their users. Eager imports here would pull the
chunker and tiktoken into every process that only needs a chunk model (the API does).
"""
