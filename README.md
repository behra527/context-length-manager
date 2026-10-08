# Context Length Manager

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.65.0-FF4B4B?logo=streamlit\&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-63%20Passed-2EA44F?logo=pytest\&logoColor=white)
![Coverage](https://img.shields.io/badge/Coverage-97%25-2EA44F)
![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-111827)

> A token-aware context management system for LLM applications, supporting budget calculation, overflow detection, truncation, chunking, and LLM-powered summarization.

## Overview

Large Language Model applications must operate within finite context windows. Sending oversized prompts can cause request failures, excessive cost, latency, or loss of important information.

**Context Length Manager** provides a reusable layer for managing this problem before context is sent to an LLM.

The system:

* Counts input tokens using `tiktoken`
* Calculates the available input-token budget
* Reserves output tokens and a safety margin
* Detects context overflow
* Applies configurable optimization strategies
* Validates the optimized context
* Tracks token usage and overflow metrics
* Integrates LLM-based summarization through OpenRouter

## Architecture

```text
                    ┌─────────────────────┐
                    │    Streamlit UI     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   ContextManager    │
                    │   Orchestration      │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
      ┌────────────┐    ┌────────────┐    ┌────────────┐
      │ Tokenizer  │    │   Budget   │    │  Metrics   │
      └────────────┘    └────────────┘    └────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Strategy Layer     │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
       ┌───────────┐     ┌───────────┐     ┌──────────────┐
       │ Truncate  │     │  Chunking │     │ Summarize    │
       └───────────┘     └───────────┘     └──────┬───────┘
                                                   │
                                                   ▼
                                          ┌─────────────────┐
                                          │ OpenRouter LLM  │
                                          └─────────────────┘
```

## Context Budgeting

The system does not treat the model's context window as entirely available for input.

The usable input budget is:

```text
max_input_tokens =
    context_window
    - max_output_tokens
    - safety_margin
```

Example:

```text
Context Window   = 8,192
Output Budget    = 1,000
Safety Margin    = 200
--------------------------------
Input Budget     = 6,992 tokens
```

This prevents the application from consuming the entire context window before accounting for model output and operational safety.

## Optimization Strategies

### 1. Truncation

Keeps the first N tokens that fit within the available budget.

```text
10,000 tokens
      │
      ▼
Keep first 6,992
      │
      ▼
Valid context
```

**Advantages**

* Fast
* Deterministic
* No additional API call
* No additional LLM cost

**Trade-off**

Information near the end of the context may be lost.

### 2. Chunking

Splits oversized input into token-based chunks with optional overlap.

Example:

```text
Chunk Size = 1,000
Overlap    =   100
```

Useful for:

* Large documents
* RAG pipelines
* Independent document processing
* Retrieval workflows

### 3. Summarization

Uses an LLM to compress oversized context.

```text
Large Context
      │
      ▼
OpenRouter
      │
      ▼
Generated Summary
      │
      ▼
Token Validation
      │
      ▼
Optimized Context
```

The generated summary is token-counted again and rejected if it exceeds the requested budget.

## Engineering Design

### Strategy Pattern

Optimization strategies are isolated from the orchestration layer:

```text
ContextManager
    ├── TruncationStrategy
    ├── ChunkingStrategy
    └── SummarizationStrategy
```

This allows additional strategies to be introduced without rewriting the core manager.

### Provider Abstraction

LLM summarization is defined through a `Summarizer` protocol.

```python
class Summarizer(Protocol):
    def summarize(self, text: str) -> str:
        ...
```

The current implementation is:

```text
Summarizer
     │
     ▼
OpenRouterSummarizer
     │
     ▼
OpenAI SDK
     │
     ▼
OpenRouter API
```

The core context-management system therefore remains independent of the LLM provider.

### Dependency Injection

The summarization provider is injected into the summarization strategy.

This enables:

* Unit testing without API calls
* Provider replacement
* Cleaner architecture
* Separation between business logic and infrastructure

## Project Structure

```text
context-length-manager/
│
├── app/
│   ├── budget.py
│   ├── config.py
│   ├── manager.py
│   ├── metrics.py
│   ├── models.py
│   ├── tokenizer.py
│   │
│   ├── llm/
│   │   └── openrouter.py
│   │
│   └── strategies/
│       ├── chunking.py
│       ├── summarization.py
│       └── truncation.py
│
├── tests/
├── scripts/
│   └── test_real_openrouter.py
│
├── streamlit_app.py
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
└── README.md
```

## Technology Stack

| Technology    | Purpose                         |
| ------------- | ------------------------------- |
| Python 3.13   | Application development         |
| tiktoken      | Token counting and encoding     |
| Streamlit     | Interactive UI                  |
| OpenAI SDK    | LLM client                      |
| OpenRouter    | LLM provider                    |
| python-dotenv | Environment configuration       |
| pytest        | Automated testing               |
| pytest-cov    | Coverage analysis               |
| dataclasses   | Typed configuration and results |
| Protocol      | Provider abstraction            |

## Installation

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/context-length-manager.git
cd context-length-manager
```

### 2. Create virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

## Environment Configuration

Create `.env`:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

The API key is loaded through environment variables and is not hardcoded in the application.

`.env` is excluded through `.gitignore`.

## Run the Application

```powershell
streamlit run streamlit_app.py
```

The UI allows users to:

1. Select a model
2. Enter context
3. Analyze token usage
4. Select an optimization strategy
5. Configure chunk parameters when required
6. Optimize oversized context
7. Inspect the resulting token metrics

If the input already fits within the available budget, the system leaves it unchanged.

## Testing

Run the complete test suite:

```powershell
pytest -v
```

Current verified result:

```text
63 passed
```

Run coverage:

```powershell
pytest --cov=app --cov-report=term-missing
```

Current verified coverage:

```text
97%
```

The test suite covers:

* Token counting
* Token encoding and decoding
* Budget validation
* Model configuration
* Context metrics
* Overflow detection
* Truncation
* Chunking
* Chunk overlap
* Summarization
* Summary budget validation
* Context manager orchestration
* OpenRouter integration behavior
* API key validation
* Edge cases and invalid configuration

## Real LLM Integration

A separate integration script verifies the actual OpenRouter connection:

```powershell
python -m scripts.test_real_openrouter
```

This performs a real LLM request and generates a summary.

Unit tests use dependency injection and do not require real API calls.

## Security

The project follows basic application security practices:

* API keys are loaded from environment variables
* Secrets are excluded from Git
* `.env.example` contains only placeholders
* Missing credentials fail explicitly
* Provider-specific credentials are isolated from business logic

## Limitations

Current limitations include:

* Model context limits are maintained in a local registry
* Truncation currently preserves the beginning of the input
* Chunking is token-based rather than semantic
* Chunk outputs are not automatically aggregated
* Summarization requires an external LLM call
* Strategy selection is currently manual

## Future Improvements

Planned improvements include:

* Automatic strategy selection
* Priority-aware context preservation
* Hybrid truncation + summarization
* Semantic chunking
* Cost and latency tracking
* REST API
* Additional LLM providers
* RAG integration
* Context prioritization based on relevance

## Key Engineering Concepts Demonstrated

This project demonstrates practical implementation of:

* LLM context-window management
* Token-aware application design
* Context budgeting
* Overflow handling
* Strategy pattern
* Dependency injection
* Provider abstraction
* LLM integration
* Configuration management
* Error handling
* Automated testing
* Test coverage
* Secure API-key management
* RAG-oriented context optimization




