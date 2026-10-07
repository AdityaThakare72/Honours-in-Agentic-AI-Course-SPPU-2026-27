# Honours Course in Agentic AI (SPPU 2026–27)

Course materials, lecture notes, and practical laboratory notebooks for the **Honour Course in Agentic AI** under the Board of Studies (Computer Engineering), Faculty of Science and Technology, **Savitribai Phule Pune University (SPPU)** (NEP-2020 Compliant Curriculum, Academic Year 2026–27).

---

## About the Programme

The Honours Programme in Agentic AI is an 18-credit specialization spanning Semesters V through VIII for undergraduate students in Computer Engineering (CE), Information Technology (IT), Artificial Intelligence & Machine Learning (AI&ML), Data Science (DS), and allied disciplines. It covers foundation models, prompt engineering, retrieval-augmented generation (RAG), vector databases, autonomous agents, multi-agent orchestration, and production AI prototyping.

### Curriculum Structure

| Year & Semester | Course Code | Course Name | Theory Credits | Practical Credits | Total Credits |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **T.E. – Sem V** | `HAAI301COM` | Applied Intelligent Systems | 3 | – | 3 |
| **T.E. – Sem V** | `HAAI302COM` | Intelligent System Laboratory | – | 1 | 1 |
| **T.E. – Sem VI** | `HAAI351COM` | Agentic AI | 3 | – | 3 |
| **T.E. – Sem VI** | `HAAI352COM` | Agentic AI Lab | – | 1 | 1 |
| **B.E. – Sem VII** | `HAAI401COM` | AI Workflow & Orchestration | 3 | – | 3 |
| **B.E. – Sem VII** | `HAAI402COM` | AI Workflow & Orchestration Lab | – | 1 | 1 |
| **B.E. – Sem VIII** | `HAAI451COM` | Advanced AI Prototyping | 3 | – | 3 |
| **B.E. – Sem VIII** | `HAAI452COM` | AI Prototyping Lab | – | 1 | 1 |
| **B.E. – Sem VIII** | `HAAI452COM` | Honours Project | – | 2 | 2 |
| **Total** | | | **12** | **6** | **18** |

---

## Semester V: Applied Intelligent Systems (`HAAI301COM` & `HAAI302COM`)

### Theory Syllabus Overview (`HAAI301COM`)
- **Unit I – LLMs & Foundation Models:** Language models, Transformer architecture, tokenization (BPE, WordPiece), pre-training, fine-tuning (LoRA/PEFT), evaluation and benchmarking.
- **Unit II – Prompt Engineering & RAG:** Zero-shot, few-shot, Chain-of-Thought (CoT) prompting, Retrieval-Augmented Generation (RAG) foundations and architectures, RAG evaluation (RAGAs), prompt safety and guardrails.
- **Unit III – Vector Databases & Embedding:** Dense text embeddings, similarity metrics, vector database architectures (FAISS, Chroma, Pinecone, Qdrant), semantic search systems, domain adaptation.
- **Unit IV – Multimodal AI (Vision, Audio, Text):** Vision-Language Models (VLMs), image generation and editing, speech recognition (ASR) and synthesis (TTS), multimodal RAG.
- **Unit V – AI APIs & Integration Patterns:** AI API ecosystems, orchestration frameworks, tool use and function calling, agentic workflows, deployment, security, ethics, and governance.

---

## Repository Contents

### Practical & Demo Notebooks
| File | Description |
| :--- | :--- |
| `Experiment_03_Advanced_Prompt_Engineering.ipynb` | Implementation and evaluation of Zero-shot, Few-shot, and Chain-of-Thought (CoT) prompting strategies across NLP tasks. |
| `Experiment_04_RAG_LangChain_Chroma_RAGAs.ipynb` | End-to-end Retrieval-Augmented Generation (RAG) pipeline using LangChain and ChromaDB over PDF documents, evaluated with RAGAs metrics (faithfulness, answer relevancy, context precision). |
| `Experiment_05_Semantic_Search_FAISS_TFIDF.ipynb` | Semantic search engine using SentenceTransformers and FAISS approximate nearest-neighbor indexing, benchmarked against traditional keyword-based TF-IDF retrieval. |
| `Experiment_Prompt_Safety_and_Guardrails.ipynb` | Practical demonstration of prompt injection vulnerabilities, input/output validation, and safety guardrails for LLM applications. |

### Syllabus, Lecture Notes & Reference Files
| File | Description |
| :--- | :--- |
| `Honours_in_Agenetic_AI.pdf` | Official SPPU NEP-2020 Honours Course in Agentic AI syllabus booklet (Semesters V–VIII). |
| `Prompt_Safety_and_Guardrails_Lecture.pdf` | Lecture notes on prompt safety, adversarial attacks, and guardrail architectures. |
| `Semantic_Search_FAISS_TFIDF_Lecture.pdf` | Lecture notes on dense embeddings, similarity search with FAISS, and comparison with lexical TF-IDF search. |
| `Project_Hail_Mary_1_to_50.pdf` | Sample PDF document corpus used for document ingestion and retrieval in the RAG experiment. |

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AdityaThakare72/Honours-in-Agentic-AI-Course-SPPU-2026-27.git
   cd Honours-in-Agentic-AI-Course-SPPU-2026-27
   ```

2. **Configure environment variables:**
   Copy `.env.example` to `.env` and add your Gemini API key:
   ```bash
   cp .env.example .env
   ```
   Update `.env`:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```

3. **Run the notebooks:**
   Open any `.ipynb` notebook in Jupyter Lab, Jupyter Notebook, or VS Code and execute the cells sequentially.
