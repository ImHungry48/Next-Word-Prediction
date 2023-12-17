# Next-Word-Prediction

# Sentence Mutator

The `SentenceMutator` class provides a simple and customizable mechanism for introducing typos and mutations into sentences. It supports five types of mutations: swapping characters, substituting characters, deleting characters, inserting characters, and changing the case of characters. The mutations are applied with a specified probability (`typo_chance`) to each character in a word.

## Usage

```python
from sentence_mutator import SentenceMutator

# Example usage
texts = ["This is a sample sentence.", "Another example for testing."]
mutator = SentenceMutator(texts, typo_chance=0.005)

sentence = "This is a test sentence."
mutated_sentence = mutator.mutate_sentence(sentence)

print(f"Original: {sentence}")
print(f"Mutated: {mutated_sentence}")
```

# Sentence Dataset

The `SentenceDataset` class is a PyTorch Dataset implementation that facilitates the creation of datasets for transformer-based models. It tokenizes input sentences and optionally applies sentence augmentation using the `SentenceMutator`. The tokenized sentences are then padded or truncated to a specified maximum length.

## Usage

```python
from sentence_dataset importSentenceDataset
from transformers import GPT2Tokenizer

# Example usage
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
max_length = 256

texts = ["This is a sample sentence.", "Another example for testing."]
dataset = SentenceDataset(texts, tokenizer, max_length, augmentation=True, typo_chance=0.005)

# Access tokenized sentences
tokenized_sentence = dataset.tokenized_sentences[0]
print(tokenized_sentence)
```

# TransformerModel

The `TransformerModel` class is a basic implementation of a transformer-based language model using PyTorch. It includes a positional encoding layer, a transformer encoder, and linear layers for embedding and output.

## Usage

```python
from transformer_model import TransformerModel

# Example usage
model = TransformerModel(
    ntokens=ntokens, 
    max_length=max_length, 
    d_model=d_model, 
    nhead=nhead, 
    nhid=nhid, 
    nlayers=nlayers)
```

# Training

The training script demonstrates how to train the `TransformerModel` using a simple dataset. It utilizes PyTorch's DataLoader for efficient batching and includes a sample training loop.

# Requirements
- [PyTorch](https://pytorch.org/)
- [Transformers](https://github.com/huggingface/transformers)
- [NLTK](https://www.nltk.org/)

Install the required packages using:
```bash
pip install torch transformers nltk
```