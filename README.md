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

## Models and specifics

Voyage currently provides three embedding models. All models currently have context length = 4096 (tokens) and embedding dimension = 1024. 

- `voyage-01`: the default choice with the best retrieval quality (e.g., better than OpenAI embedding models — see [blog post](https://blog.voyageai.com/2023/10/29/voyage-embeddings/) for details.) 
- `voyage-lite-01`: 2x faster inference than `voyage-01` with nearly the same retrieval quality. 
- `voyage-lite-01-instruct`: tweaked on top of `voyage-lite-01` for classification and clustering tasks, which are the only recommended use cases.

More advanced and specialized models are coming soon and please contact [contact@voyageai.com](mailto:contact@voyageai.com) for early access.

- `voyage-xl-01`: coming soon
- `voyage-code-01`: coming soon
- `voyage-finance-01`: coming soon

### Functions

he core functions are `get_embedding` that takes a single document (or query), and `get_embeddings`, which allows a batch of documents or queries.  Before using the library, please first register for a [Voyage API key](#authentication-with-api-keys).

> `get_embedding(text, model, input_type=None)` [:link:](https://github.com/voyage-ai/voyageai-python/blob/d95621a0f837a791912945edeeeae47325c5d602/voyageai/embeddings.py#L63)

- **Parameters**
    - **text** - A single document/query as a string, such as `"I like cats"` .
    - **model** - Name of the model. Options: `voyage-01`, `voyage-lite-01`.
    - **input_type** - Type of the input text. Defalut to `None`, meaning the type is unspecified. Other options:  `query`, `document`.
- **Returns**
    - An embedding vector (a list of floating-point numbers) for the document.


> `get_embeddings(list_of_text, model, input_type=None)` [:link:](https://github.com/voyage-ai/voyageai-python/blob/d95621a0f837a791912945edeeeae47325c5d602/voyageai/embeddings.py#L80)

- **Parameters**
    - **list_of_text** - A list of documents as a list of strings, such as  `["I like cats", "I also like dogs"]`. The length of the list is at most 64. (Each Voyage API request takes at most 8 strings. This function makes one API request when `len(list_of_text)<=8`, and makes multiple API requests in parallel when `8<len(list_of_text)<=64`.)
    - **model** - Name of the model. Options: `voyage-01`, `voyage-lite-01`.
    - **input_type** - Type of the input text. Defalut to `None`, meaning the type is unspecified. Other options: `query`, `document`.
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
embeddings = get_embeddings(documents, model="voyage-01", input_type="document")
```
