"""Shared helpers for notebooks 02 and 03: a Gemini chat model and a small on-disk response cache.

The cache stores every LLM response in llm_cache.json, keyed by the exact prompt and model settings.
Re-running a notebook then costs no API calls and gives the same answers. Delete the file to call the API afresh.
"""
import os, json, hashlib, warnings
from dotenv import load_dotenv, find_dotenv
from langchain_core.caches import BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.load import dumpd, load
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")                 # silence LangChain beta/pending-deprecation notices from load()
load_dotenv(find_dotenv(usecwd=True))
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class JsonFileCache(BaseCache):
    """LangChain cache that keeps responses in one JSON file."""

    def __init__(self, path="llm_cache.json"):
        self.path = path
        self.data = json.load(open(path)) if os.path.exists(path) else {}

    def _key(self, prompt, llm_string):
        return hashlib.sha256((llm_string + "||" + prompt).encode()).hexdigest()

    def lookup(self, prompt, llm_string):
        item = self.data.get(self._key(prompt, llm_string))
        return [load(g, allowed_objects="core") for g in item] if item else None

    def update(self, prompt, llm_string, return_val):
        self.data[self._key(prompt, llm_string)] = [dumpd(g) for g in return_val]
        json.dump(self.data, open(self.path, "w"))

    def clear(self, **kwargs):
        self.data = {}
        json.dump(self.data, open(self.path, "w"))


def enable_cache(path="llm_cache.json"):
    cache = JsonFileCache(path)
    set_llm_cache(cache)
    return cache


def get_llm(model="gemini-2.5-flash-lite", temperature=0.0):
    """Gemini through its OpenAI-compatible endpoint, wrapped as a LangChain chat model."""
    return ChatOpenAI(model=model, temperature=temperature, api_key=os.environ["GEMINI_API_KEY"],
                      base_url=GEMINI_URL, timeout=60, max_retries=4)


# ---- the RAG prompt and context formatter used in notebooks 02 and 03 ----
from langchain_core.prompts import ChatPromptTemplate

SYSTEM = (
    "You answer questions about research papers using ONLY the context provided. "
    "If the context does not contain the answer, reply exactly: I don't know based on the provided documents. "
    "Answer in at most three sentences."
)
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def format_docs(docs):
    """One block per chunk, labelled with a number, file name and page."""
    return "\n\n".join(
        f"[{i}] ({d.metadata['source'].split('/')[-1]}, p.{d.metadata['page'] + 1})\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )
