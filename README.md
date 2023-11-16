# Voyage Python Library

[Voyage AI](https://www.voyageai.com) provides cutting-edge embedding/vectorizations models.

Embedding models are neural net models (e.g., transformers) that convert unstructured and complex data, such as documents, images, audios, videos, or tabular data, into numerical vectors that capture their semantic meanings. These vectors serve as representations/indices for datapoints and are an essential building blocks for semantic search and retrieval-augmented generation stack (RAG), which is the dominating approach for domain-specific or company-specific chatbots. 

Voyage AI provides API endpoints for embedding models that take in your data (e.g., documents or queries) and return their embeddings. The embedding models are a modular component that can used with any other components in the RAG stack, such as any vectorDB and any generative LLM.

Voyage’s embedding models are **state-of-the-art** in retrieval accuracy. Please read our announcing [blog post](https://blog.voyageai.com/2023/10/29/voyage-embeddings/) for details.  Please also check out a high-level [introduction](https://www.pinecone.io/learn/retrieval-augmented-generation/) of embedding models, semantic search, and RAG, and our step-by-step [quickstart tutorial](https://docs.voyageai.com/tutorials/) on implementing a minimalist RAG chatbot using Voyage embeddings.

The Voyage Python library provides convenient access to the Voyage API (see our [documentation](https://docs.voyageai.com)).

## Installation
Use `pip` to install the package:
```bash
pip install voyageai
```

Test the installation by running this command:

```bash
python -c "import voyageai"
```

## Authentication with API keys

Voyage AI utilizes API keys to monitor usage and manage permissions. To obtain your key, first create an account by clicking the "SIGN IN" button on our [homepage](https://www.voyageai.com). Once signed in, access your API key by clicking "View API key" in the dashboard.

Your API key is supposed to be secret -- please avoid sharing it or exposing it in browsers or apps. Please store your API key securely for future use, e.g., via the following bash command. 

```bash
export VOYAGE_API_KEY = '[ Your VOYAGE API key ]'
```

Alternatively, the API key can be set in python (after you install the package):

```python
import voyageai
voyageai.api_key = "[ Your VOYAGE API key ]"
```

## Voyage Embeddings

### Models and specifics

Voyage offers two embedding model options: **`voyage-01`** and **`voyage-lite-01`**. The former provides the best quality and the latter is optimized for inference-time efficiency.  More advanced and specialized models are coming soon and please contact [contact@voyageail.com](mailto:contact@voyageail.com) for early access.

| Model Name | Context Length (tokens) | Embedding Dim. | Latency | Quality |
| --- | --- | --- | --- | --- |
| `voyage-01` | 4096 | 1024 | ++++ | ++++ |
| `voyage-lite-01` | 4096 | 1024 | +++ | +++ |
| `voyage-lite-01-instruct` | 4096 | 1024 | +++ | +++ |
| `voyage-xl-01` | coming soon  |  |  |  |
| `voyage-code-01` | coming soon |  |  |  |
| `voyage-finance-01` | coming soon |  |  |  |

### Functions

The core functions are `get_embedding()` that takes a single document (or query), and `get_embeddings()` , which allows a batch of documents or queries.  Before using the library, please first register for a [Voyage API key](https://docs.voyageai.com/install/).

> `get_embedding(text, model)` [🔗](https://github.com/voyage-ai/voyageai-python/blob/main/voyageai/embeddings.py#L12)

- **Parameters**
    - **text** - A single document/query as a string, such as `"I like cats"` .
    - **model** - Name of the model. Options: `"voyage-01"`, `"voyage-lite-01"`.
    - **input_type** - Type of the input text. Defalut to None, meaning the type is unspecified. Other options include: "query", "document".
- **Returns**
    - An embedding vector (a list of floating-point numbers) for the document.


> `get_embeddings(list_of_text, model)` [🔗](https://github.com/voyage-ai/voyageai-python/blob/main/voyageai/embeddings.py#L22)

- **Parameters**
    - **list_of_text** - A list of documents as a list of strings, such as  `["I like cats", "I also like dogs"]`. The length of the list is at most 8. This function only makes one API call, which takes a list of at most 8 strings.
    - **model** - Name of the model. Options: `"voyage-01"`, `"voyage-lite-01"`.
    - **input_type** - Type of the input text. Defalut to None, meaning the type is unspecified. Other options include: "query", "document".
- **Returns**
    - A list of embedding vectors.

#### **Example usage**

Given a list of documents, obtain the embeddings from Voyage Python package. 

```python
import voyageai 
from voyageai import get_embeddings

voyageai.api_key = "[ Your Voyage API KEY ]"  # add you Voyage API KEY

documents = [
    "The Mediterranean diet emphasizes fish, olive oil, and vegetables, believed to reduce chronic diseases.",
    "Photosynthesis in plants converts light energy into glucose and produces essential oxygen.",
    "20th-century innovations, from radios to smartphones, centered on electronic advancements.",
    "Rivers provide water, irrigation, and habitat for aquatic species, vital for ecosystems.",
    "Apple’s conference call to discuss fourth fiscal quarter results and business updates is scheduled for Thursday, November 2, 2023 at 2:00 p.m. PT / 5:00 p.m. ET.",
    "Shakespeare's works, like 'Hamlet' and 'A Midsummer Night's Dream,' endure in literature."
]

# Embed the documents
embeddings = get_embeddings(documents, model="voyage-01")
```
