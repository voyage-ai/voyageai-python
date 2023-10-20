# VoyageAI Python Library
The VoyageAI Python library provides convenient access to the VoyageAI API.

## Installation
Use `pip` to install the package:
```bash
pip install voyageai
```

## Usage
First, import the libary at the top of a file:
```python
import voyageai
```
The library needs to be configured with your VoyageAI API key. Either set it as the `VOYAGEAI_API_KEY` environment variable before using the library:
```bash
export VOYAGEAI_API_KEY="[ Your VoyageAI API key ]"
```
Or set openai.api_key to its value:
```python
voyageai.api_key = "[ Your VoyageAI API key ]"
```

### Embedding
```python
text = "sample text"
model = "voyage-api-v0"
embedding = voyageai.Embedding.create(input=text, model=model)["data"][0]["embedding"]
```
