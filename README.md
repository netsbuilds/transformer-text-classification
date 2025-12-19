# Transformer Text Classification (AG News)

This project fine-tunes a pretrained DistilBERT model for multi-class
text classification on the AG News dataset.

## Model
- DistilBERT (distilbert-base-uncased)
- Classification head fine-tuned on 4 news categories

## Training
- 5,000 training samples
- 2 epochs
- Cross-entropy loss
- AdamW optimizer

## Results
- Accuracy: ~90%
- Weighted F1: ~90%

This project was built alongside Stanford CS224N coursework to
understand practical transformer fine-tuning.
