# DistilBERT Question Answering

This project fine-tunes DistilBERT for question answering on the SQuAD dataset.

## Requirements

- transformers
- torch
- datasets

Install the dependencies:

```bash
pip install -r requirements.txt
```

To train the BERT model on the IMDB dataset, run:
```bash
python scripts/train.py
```

To evaluate the fine-tuned model on the test set, run:
```bash
python scripts/evaluate.py
```
