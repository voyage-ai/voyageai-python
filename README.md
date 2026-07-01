# Voyage Python Library

[Voyage AI](https://www.voyageai.com) provides cutting-edge embedding and rerankers.

Embedding models are neural net models (e.g., transformers) that convert unstructured and complex data, such as documents, images, audios, videos, or tabular data, into dense numerical vectors (i.e. embeddings) that capture their semantic meanings. These vectors serve as representations/indices for datapoints and are essential building blocks for semantic search and retrieval-augmented generation (RAG), which is the predominant approach for domain-specific or company-specific chatbots and other AI applications.

Rerankers are neural nets that output relevance scores between a query and multiple documents. It is common practice to use the relevance scores to rerank the documents initially retrieved with embedding-based methods (or with lexical search algorithms such as BM25 and TF-IDF). Selecting the highest-scored documents refines the retrieval results into a more relevant subset.

Voyage AI provides API endpoints for embedding and reranking models that take in your data (e.g., documents, queries, or query-document pairs) and return their embeddings or relevance scores. Embedding models and rerankers, as modular components, seamlessly integrate with other parts of a RAG stack, including vector stores and generative Large Language Models (LLMs).

Voyage AI’s embedding models and rerankers are **state-of-the-art** in retrieval accuracy. Please read our announcing [blog post](https://blog.voyageai.com/2023/10/29/voyage-embeddings/) for details.  Please also check out a high-level [introduction](https://www.pinecone.io/learn/retrieval-augmented-generation/) of embedding models, semantic search, and RAG, and our step-by-step [quickstart tutorial](https://docs.voyageai.com/docs/quickstart-tutorial) on implementing a minimalist RAG chatbot using Voyage model endpoints.

## Local models (`voyage-4-nano`)

`voyage-4-nano` can run **locally**, without an API key, via
[sentence-transformers](https://www.sbert.net/). Install the optional extra:

```bash
pip install "voyageai[local]"
```

Requirements:

- **Python 3.10+** (the local model's code uses 3.10+ typing syntax). On older
  Python the extra is skipped and local models are unavailable.
- The extra pulls in `torch` and `sentence-transformers`; a pure-API install of
  `voyageai` never imports them, so API-only users pay no import cost.

Usage is identical to the hosted API — just pass the local model name; no API key
is required:

```python
import voyageai

vo = voyageai.Client()  # no API key needed for local models
result = vo.embed(["Hello, world!"], model="voyage-4-nano", input_type="document")
print(result.embeddings[0])
```

The local path mirrors the hosted API's behavior (input validation, unit-norm
Matryoshka embeddings at every dimension, token accounting, and the 1000-input
batch limit). `int8`/`uint8` output dtypes are the one exception — they require
fixed calibration ranges we don't ship locally yet and raise `NotImplementedError`;
use `float`/`float32` or `binary`/`ubinary` instead.

### [Voyage AI Official Documentation](https://docs.voyageai.com)
