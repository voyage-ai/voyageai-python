# Voyage Python Library

[Voyage AI](https://www.voyageai.com) provides cutting-edge embedding/vectorizations models.

Embedding models are neural net models (e.g., transformers) that convert unstructured and complex data, such as documents, images, audios, videos, or tabular data, into numerical vectors that capture their semantic meanings. These vectors serve as representations/indices for datapoints and are an essential building blocks for semantic search and retrieval-augmented generation stack (RAG), which is the dominating approach for domain-specific or company-specific chatbots. 

Voyage AI provides API endpoints for embedding models that take in your data (e.g., documents or queries) and return their embeddings. The embedding models are a modular component that can used with any other components in the RAG stack, such as any vectorDB and any generative LLM.

Voyage’s embedding models are **state-of-the-art** in retrieval accuracy. Please read our announcing [blog post](https://blog.voyageai.com/2023/10/29/voyage-embeddings/) for details.  Please also check out a high-level [introduction](https://www.pinecone.io/learn/retrieval-augmented-generation/) of embedding models, semantic search, and RAG, and our step-by-step [quickstart tutorial](https://docs.voyageai.com/tutorials/) on implementing a minimalist RAG chatbot using Voyage embeddings.

The Voyage Python library provides convenient access to the Voyage API (see our [documentation](https://docs.voyageai.com)).

## Authentication with API keys

Voyage AI utilizes API keys to monitor usage and manage permissions. To obtain your key, please sign in with your Voyage AI account and click the "Create new API key" button in the [dashboard](https://dash.voyageai.com).

Your API key is supposed to be secret -- please avoid sharing it or exposing it in browsers or apps. Please store your API key securely for future use. 


## Install Voyage Python library

You can interact with the API through HTTP requests from any language. For Python users, we offer an official library which can be installed via `pip` :
```bash
pip install -U voyageai
```
We recommend using the `-U` or `--upgrade` option to ensure you are installing the latest version of the package. This helps you access the most recent features and bug fixes.

After installation, you can test it by running:
```bash
python -c "import voyageai"
```
The installation is successful if this command runs without any errors.


## Models and Specifics

Voyage currently provides the following embedding models.

| Model | Context Length (tokens) | Embedding Dimension | Description |
| :---- | :---: | :---: | --- |
| `voyage-02` | 4000 | 1024 | Default embedding model with the best retrieval quality (e.g., better than OpenAI embedding models — see [blog post](https://blog.voyageai.com/2023/10/29/voyage-embeddings/) for details). |
| `voyage-code-02` | 16000 | 1536 | Code embedding model optimized for code retrieval. |

More advanced and specialized models are coming soon and please email [contact@voyageai.com](mailto:contact@voyageai.com) for early access.

- `voyage-multilingual-02`: coming soon
- `voyage-finance-02`: coming soon
- `voyage-healthcare-02`: coming soon
- `voyage-large-02`: coming soon

### Deprecated

The following models are our first-generation models, which are still accessible from our API. We recommend to use the new models above for better quality and efficiency.

| Model | Context Length (tokens) | Embedding Dimension | Description |
| :---- | :---: | :---: | --- |
| `voyage-01` | 4000 | 1024 | [_Deprecated_] Our v1 embedding model. |
| `voyage-lite-01` | 4000 | 1024 | [_Deprecated_] 2x faster inference than `voyage-01` with nearly the same retrieval quality. |
| `voyage-lite-01-instruct` | 4000 | 1024 | [_Deprecated_] Tweaked on top of `voyage-lite-01` for classification and clustering tasks, which are the only recommended use cases. |

## Create Embeddings

We provide the `voyageai.Client` class as a Python interface for interacting with our API server. This client enables you to use our API for converting texts into embeddings and accessing utility functions, including tokenization.

`class voyageai.Client` 

**Parameters**

- **api_key** (str, optional, defaults to `None`) - Voyage API key. If `None`, the client will search for the API key in the following order:
    * `voyageai.api_key_path`, path to the file containing the key;
    * `voyageai.api_key`;
    * environment variable `VOYAGE_API_KEY`.

<!-- - **max_retries** (int, optional, defaults to 3) - Maximum number of retries if an API call fails.
- **timeout** (int, optional, defaults to 120) - Timeout in seconds for the API request. -->

`embed(texts : List[str], model : str = "voyage-02", input_type : Optional[str] = None, truncation : Optional[bool] = None)`

**Parameters** 

- **texts** (List[str]) - A list of texts as a list of strings, such as `["I like cats", "I also like dogs"]`. Currently, the maximum length of the list is 128, and there is also a limit on the total number of tokens for input texts: 320K for `voyage-02` and 120K for `voyage-code-02`.
- **model** (str) - Name of the model. Recommended options: `voyage-02` (default), `voyage-code-02`.
- **input_type** (str, optional, defaults to `None`) - Type of the input text. Defalut to `None`, meaning the type is unspecified. Other options:  `query`, `document`.
- **truncation** (bool, optional, defaults to `None`) - Whether to truncate the input texts to fit within the context length limit of our embedding models.
    - If `True`, over-length input texts will be truncated to fit within the context length, before encoded by the embedding model.
    - If `False`, an error will be raised if any given texts exceeds the context length.
    - If not specified (defaults to `None`), we will truncate the input text before sending it to the embedding model if it slightly exceeds the context window length. If it significantly exceeds the context window length, an error will be raised.

**Returns**

- A `voyageai.EmbeddingsObject`, containing the following attributes:
    - **embeddings** (List[List[float]]) - A list of embeddings for the corresponding list of input texts, where each embedding is a vector represented as a list of floats.
    - **total_tokens** (int) - The total number of tokens for the input texts.

**Example usage**

Given a list of texts, obtain the embeddings from Voyage Python package. First, set the API key as an environment variable `export VOYAGE_API_KEY="api-key"`.

```python
import os
import voyageai

vo = voyageai.Client(
    api_key=os.environ.get("VOYAGE_API_KEY"),
)

texts = [
    "The Mediterranean diet emphasizes fish, olive oil, ...",
    "Photosynthesis in plants converts light energy into ...",
    "20th-century innovations, from radios to smartphones ...",
    "Rivers provide water, irrigation, and habitat for ...",
    "Apple’s conference call to discuss fourth fiscal ...",
    "Shakespeare's works, like 'Hamlet' and ...",
]

# Embed the documents
result = vo.embed(texts, model="voyage-02", input_type="document")
print(result.embeddings)
```
        
Output
           
```
[
    [0.005262283142656088, -0.0003730150347109884, -0.025250812992453575, ...],
    [-0.008110595867037773, 0.012202565558254719, -0.05196673423051834, ...],
    [-0.007480636239051819, 0.014599844813346863, -0.022662457078695297, ...],
    [0.004256163723766804, 0.024158479645848274, -0.03241389989852905, ...],
    [0.00873539038002491, 0.0037946170195937157, -0.026066618040204048, ...],
    [0.018981195986270905, 0.02546825259923935, -0.05191393196582794, ...]
]
```


`tokenize(texts : List[str])`

**Parameters**

- **texts** (List[str]) - A list of texts to be tokenized. Currently, all Voyage embedding models use the same tokenizer (see [here](#tokenization) for details).

**Returns**
       
- A list of [`tokenizers.Encoding`](https://huggingface.co/docs/tokenizers/main/en/api/encoding#encoding), each represents the tokenized encoding of a document.

**Example usage**
        
```python
tokenized = vo.tokenize(texts)
for i in range(len(texts)):
    print(tokenized[i].tokens)
```
        
Output

```bash
['<s>', '▁The', '▁Mediter', 'rane', 'an', '▁di', 'et', '▁emphas', 'izes', '▁fish', ',', '▁o', 'live', '▁oil', ',', '▁...']
['<s>', '▁Ph', 'otos', 'yn', 'thesis', '▁in', '▁plants', '▁converts', '▁light', '▁energy', '▁into', '▁...']
['<s>', '▁', '2', '0', 'th', '-', 'century', '▁innov', 'ations', ',', '▁from', '▁rad', 'ios', '▁to', '▁smart', 'ph', 'ones', '▁...']
['<s>', '▁R', 'ivers', '▁provide', '▁water', ',', '▁ir', 'rig', 'ation', ',', '▁and', '▁habitat', '▁for', '▁...']
['<s>', '▁Apple', '’', 's', '▁conference', '▁call', '▁to', '▁discuss', '▁fourth', '▁fis', 'cal', '▁...']
['<s>', '▁Shakespeare', "'", 's', '▁works', ',', '▁like', "▁'", 'H', 'am', 'let', "'", '▁and', '▁...']
```


`count_tokens(texts : List[str])` 

**Parameters**

- **texts** (List[str]) - A list of texts to count the tokens for.

**Returns**
        
- An integer representing the total number of tokens for the input texts.

**Example usage**

```python
total_tokens = vo.count_tokens(texts)
print(total_tokens)
```

Output
```bash
86
```

