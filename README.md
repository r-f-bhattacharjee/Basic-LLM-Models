# Basic LLM Models

A clean, hands-on repository containing concise, runnable examples and utilities for working with Large Language Models (LLMs). This project serves as a practical guide to prototyping prompts, exploring OpenAI's models via LangChain, and implementing mathematical semantic search using vector embeddings.

Use these examples to understand basic LLM model output response using Langchain.

## Features & Code Walkthrough

The repository is divided into three fundamental use cases, showcasing the evolution from raw text prediction to semantic vector spaces.

### 1. Legacy Text Completion vs. Modern Chat Models
Demonstrates how to interact with different model paradigms using LangChain interfaces.
* **Completion Model (`gpt-3.5-turbo-instruct`):** Interacts with older base OpenAI models, instruction-following completion architectures using standard text orchestration.
* **Chat Model (`gpt-4o`):** Utilizes state-of-the-art conversational APIs, adjusting creativity thresholds via `temperature` and hard limits with `max_tokens`.

### 2. Semantic Search via Vector Embeddings
Demonstrates the underlying math of retrieval-augmented systems. Text sentences are mapped into high-dimensional vector spaces using OpenAI's `text-embedding-3-small` model, and ranked by conceptual relevance.

### The Mathematical Foundation

To identify the document most relevant to a query, the script computes the Cosine Similarity between the vector embeddings of the query and each document (V1 and V2):

Cosine Similarity= V1.V2/∥V1∥∥V2∥
	
Dot Product (V1.V2): Captures how closely the query and document vectors align in both direction and magnitude.
Vector Norms (∥V1∥ , ∥V2∥): Scale the vectors to remove length-based bias, allowing the comparison to focus purely on semantic similarity.
Ranking: Applies NumPy-based sorting (np.argsort) to rank similarity scores and retrieve the document with the highest value.

## Tech Stack

* **Language:** Python 3.10+
* **Orchestration:** LangChain (`langchain-openai`)
* **Mathematical Utilities:** NumPy
* **Environment Management:** Python `python-dotenv`

## Quick Start & Setup

### 1. Clone the Repository & Navigate In
```bash
git clone [https://github.com/r-f-bhattacharjee/Basic-LLM-Models.git](https://github.com/r-f-bhattacharjee/Basic-LLM-Models.git)
cd Basic-LLM-Models

```

### 2. Set Up Your Virtual Environment

```bash
python -m venv env1
source env1/bin/activate  # On Windows: .\env1\Scripts\activate

```

### 3. Install Required Dependencies

Create a `requirements.txt` file (or install manually) containing `langchain-openai`, `python-dotenv`, and `numpy`, then run:

```bash
pip install -r requirements.txt

```

### 4. Configure Environment Secrets

Create a `.env` file in the root directory of your project:

```env
 OPENAI_API_KEY=your_actual_openai_api_key_here

```

*(Note: The `.env` file is protected locally and securely ignored by `.gitignore` to prevent secret leakage.)*

---

## 📂 File Execution

To run the models or embedding search, execute the Python scripts via your terminal:

```bash
# To test LLM Base Model output responses
python llm_base.py

# To test LLM Chat Model output responses
python llm_chat.py

# To compute vector weights and run the semantic comparison
python llm_embedding.py

```

