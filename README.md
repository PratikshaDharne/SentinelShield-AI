# SentinelShield AI — Agentic Real-Time Guardrail Firewall for LLM Security

SentinelShield AI is an AI-powered security layer designed to protect Large Language Model (LLM) applications from prompt injection, jailbreak attempts, prompt leakage, and potentially harmful inputs. It analyzes incoming requests and decides whether to **Allow, Rewrite, or Block** them before they reach the LLM.

The project combines rule-based security checks, machine learning, semantic retrieval, and reinforcement learning to provide an additional layer of protection for LLM-powered applications.

## Problem Statement

LLM applications can be vulnerable to malicious prompts, attempts to extract hidden instructions, and requests that bypass intended safety restrictions. Relying only on the underlying LLM to identify these attacks can be insufficient.

SentinelShield AI addresses this problem by introducing a dedicated guardrail layer that evaluates incoming prompts before they are processed by an LLM.

## Key Features

- **Prompt Threat Detection:** Identifies prompt injection, jailbreak attempts, prompt leakage, and other suspicious inputs.
- **Intelligent Decision System:** Classifies requests into Allow, Rewrite, or Block decisions based on security checks and model predictions.
- **Controlled Prompt Rewriting:** Rewrites eligible prompts while preserving their original intent and checks the rewritten output before allowing it.
- **Semantic Retrieval:** Uses embeddings and FAISS-based retrieval to identify relevant security examples.
- **Reinforcement Learning:** Includes a learning-based policy for improving decisions using labelled feedback and offline training.
- **Secure LLM Gateway:** Routes approved prompts to Gemini, with Groq as a fallback provider.
- **Monitoring Dashboard:** Provides a Streamlit interface for monitoring requests, reviewing decisions, evaluating models, and managing API keys.
- **Privacy-Aware Auditing:** Supports security monitoring using decision metadata, risk information, and request metrics without storing raw prompts by default.

## How It Works

1. **Request:** An application sends a prompt to the SentinelShield API.
2. **Detection:** The system normalizes the input and evaluates it using security rules, classifiers, and retrieval-based evidence.
3. **Decision:** The policy determines whether the request should be allowed, rewritten, or blocked.
4. **Validation:** Rewritten prompts undergo safety and intent-preservation checks.
5. **LLM Processing:** Approved prompts are forwarded to the configured LLM provider.
6. **Monitoring:** The dashboard displays request activity, decisions, latency, and evaluation results.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn, NumPy |
| NLP & Embeddings | Sentence Transformers, MiniLM |
| Semantic Retrieval | FAISS |
| Reinforcement Learning | Q-Learning, Double DQN |
| Backend API | FastAPI |
| Dashboard | Streamlit |
| Database | MySQL |
| LLM Providers | Gemini, Groq fallback |
| Deployment | Docker, Docker Compose |
| Testing | Pytest |

## Architecture

```text
LLM Application
       |
       v
FastAPI Security Gateway
       |
       v
Input Normalization
       |
       v
Security Rules + ML Detection
       |
       v
Semantic Retrieval + Policy
       |
       v
Allow / Rewrite / Block
       |
       v
Rewrite Validation
       |
       v
Approved Prompt
       |
       v
Gemini / Groq Fallback
       |
       v
Output Validation
       |
       v
Application Response

Streamlit Dashboard
       |
       +-- Request Monitoring
       +-- Security Audit
       +-- Model Evaluation
       +-- Policy Management
       +-- API Key Management
```

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version specified by the project
- Git
- Docker Desktop, for Docker-based deployment
- API credentials for the configured LLM providers

### 1. Clone the repository

```bash
git clone https://github.com/PratikshaDharne/SentinelShield-AI.git
cd SentinelShield-AI
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure environment variables

Create your local `.env` file from the supplied example:

```powershell
Copy-Item .env.example .env
```

Configure the required passwords, secrets, and LLM API keys in `.env`. Never commit this file to GitHub.

### 4. Run the application

For Docker-based deployment, use the repository's configured Docker Compose file and environment settings. The documented default service ports are:

- **API:** `8130`
- **Dashboard:** `8630`
- **MySQL:** `3337`

For local development, follow the project's API and dashboard startup instructions in the repository.

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/guard` | Analyze a prompt and return a security decision |
| `POST /v1/rewrite` | Rewrite an eligible prompt |
| `POST /v1/chat` | Securely process a prompt through the LLM gateway |
| `POST /v1/chat/completions` | Provide an OpenAI-compatible chat interface |
| `POST /v1/guard/tool` | Evaluate agent and tool-call requests |
| `GET /v1/health` | Check API health |
| `GET /v1/metrics` | Access supported monitoring metrics |

Protected endpoints require the appropriate API key or owner authentication.

## Security and Privacy

- Raw prompt logging is disabled by default through the `LOG_RAW_PROMPTS` setting.
- Audit records can store decision metadata, risk information, hashes, and latency.
- API keys are stored in hashed form.
- Rewritten prompts are checked before they are passed to the LLM.
- Security rules and policy constraints limit which decisions the learning-based agent can make.

These mechanisms provide additional safeguards but do not guarantee protection against every possible attack.

## Testing and Evaluation

The project includes automated tests and evaluation workflows for examining detection, rewriting, decision policies, and security behavior.

Evaluation can cover:

- Attack detection and recall
- False-positive and false-negative rates
- Rewrite safety and intent preservation
- Model and policy comparisons
- Request latency and performance
- Robustness against unseen attack phrasing

Model performance depends on the dataset, training configuration, and evaluation conditions. Refer to the generated evaluation reports for measured results and limitations.

## Project Goal

SentinelShield AI aims to make LLM applications safer by placing a dedicated, measurable, and extensible security layer between users and language models.

**GitHub Repository:** https://github.com/PratikshaDharne/SentinelShield-AI
