"""LLM/RAG integration for interactive narration.

Talks to a local Ollama server (chat + embeddings) over plain HTTP — no
LLM client SDK dependency, see `ollama.py`. `prompts.py` builds the message
list each mode sends; `narrator_guidelines.md` is the shared, editable
system-prompt "constitution" (spoiler safety, RAG grounding discipline)
every mode's prompt is built on top of.
"""
