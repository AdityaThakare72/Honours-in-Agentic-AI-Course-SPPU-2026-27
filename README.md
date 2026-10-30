# HAAI302COM - Artificial Intelligence Laboratory

Ten practicals as Jupyter notebooks, one folder each. Every folder holds two to three numbered notebooks that are meant to be run in order. Each concept follows the same rhythm: a markdown cell that explains the step, a short code cell, and a markdown note on what to look for in the output. Parameters are varied side by side so you can see what each one changes.

| # | Folder | Topic | Notebooks |
|---|--------|-------|-----------|
| 1 | `Experiment_01_Tokenization_Text_Generation` | BPE and WordPiece tokenization, GPT-2 decoding parameters | 2 |
| 2 | `Experiment_02_LoRA_Fine_Tuning` | LoRA anatomy with PEFT, fine-tuning for text-to-SQL | 2 |
| 3 | `Experiment_03_Prompt_Engineering` | Zero-shot, few-shot, chain-of-thought, self-consistency | 2 |
| 4 | `Experiment_04_RAG_LangChain_Chroma_RAGAs` | Chunking, Chroma, a LangChain RAG chain, RAGAs evaluation | 3 |
| 5 | `Experiment_05_Semantic_Search_FAISS_TFIDF` | TF-IDF baseline, SentenceTransformers, FAISS indexes, hybrid search | 3 |
| 6 | `Experiment_06_Vector_DB_Qdrant_Pinecone` | Qdrant ingestion, filtered search, index configuration and recall | 3 |
| 7 | `Experiment_07_Visual_QA_Gemini_LLaVA` | Visual question answering, VQA metrics, open-source vision-language models | 3 |
| 8 | `Experiment_08_Voice_Assistant_Whisper_LLM_TTS` | Whisper speech recognition and WER, LLM and TTS, end-to-end assistant | 3 |
| 9 | `Experiment_09_LangGraph_Agent_Tools_HITL` | Tool calling, LangGraph stateful agent, human-in-the-loop checkpoints | 3 |
| 10 | `Experiment_10_FastAPI_AI_Microservice` | FastAPI service with streaming, Redis semantic cache, logging and tracing | 3 |

## Lab journal versions

The folder `Labs_Required_Farmat` holds the same ten experiments in the format required for the lab journal: Aim, Objectives, Software Requirements, Theory, Procedure, Python Program, Observation Tables, Result and Conclusion. Each experiment is a single, shorter notebook that uses high-level libraries, and the saved outputs are the ones shown in the observations. The `Experiment_NN_...` folders above contain the longer, step-by-step versions.

## Setup

1. Python 3.10 or newer. Create an environment and install the packages:
   ```
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (a free key from Google AI Studio). Language-model calls go through the OpenAI SDK pointed at Gemini's OpenAI-compatible endpoint. `.env` is git-ignored.
3. Start Jupyter and open a notebook:
   ```
   jupyter lab
   ```

Optional keys, all read from `.env` and skipped when absent: `PINECONE_API_KEY` (Experiment 6), `ELEVENLABS_API_KEY` (Experiment 8), `LANGSMITH_API_KEY` with `LANGSMITH_TRACING=true` (Experiment 10).

## Services used by some experiments

Experiment 6 needs Qdrant and Experiment 10 needs Redis. Both run locally in Docker:

```
docker run -d --name lab-qdrant -p 127.0.0.1:6333:6333 qdrant/qdrant
docker run -d --name lab-redis  -p 127.0.0.1:6379:6379 redis:7-alpine
```

Remove them afterwards with `docker rm -f lab-qdrant lab-redis`.

## Notes on running

- **Order matters inside a folder.** Later notebooks read files that earlier ones write (embeddings, caches, service code). In Experiment 10, run notebook 02 before 03.
- **Caches.** Notebooks that call Gemini keep answers in `llm_cache.json`, so re-running them is fast and works offline. Delete the file to query the model again.
- **No GPU required** for the executed parts. Cells that need one (LoRA training in Experiment 2, 4-bit LLaVA in Experiment 7) are guarded by `torch.cuda.is_available()` and are intended for a free Colab T4 or better. They are explained in the notebooks but are not run in the saved outputs.
- **Not run in the saved outputs:** the Pinecone cells (need a key), ElevenLabs and live-microphone cells (need a key and hardware), and the LangSmith cell.
- **Downloads.** The first run of a notebook downloads models and small datasets from the Hugging Face Hub (GPT-2, MiniLM, Whisper, SmolVLM, BEIR SciFact, LibriSpeech and VQA samples). Plan for a few GB of disk.
- **Experiment 10** starts its FastAPI servers as background processes from inside the notebooks, on ports 8101 and above, and stops them at the end of each notebook.

## Folder contents

Besides the notebooks, folders contain small helper modules (`*_utils.py`, `llm_setup.py`, `agent_tools.py`), the data they use, and the cache or output files that notebooks write. These are safe to delete and will be regenerated.
