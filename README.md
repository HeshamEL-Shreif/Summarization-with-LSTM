#  Summarization with LSTM

This project implements a text summarization model using a sequence-to-sequence LSTM architecture trained on the [BillSum dataset](https://huggingface.co/datasets/billsum). It uses Byte Pair Encoding (BPE) tokenization and is built with PyTorch.

##  Features

- Custom tokenizer built using Hugging Face's `tokenizers` library with BPE.
- Sequence-to-sequence LSTM model with optional bidirectionality.
- Packed padded sequence handling for efficient training.
- Custom dataset and dataloader with dynamic padding.
- Trains on the BillSum dataset for abstractive summarization.
- Evaluation with token-level accuracy tracking during training.

##  Model Architecture

- **Encoder**: Embedding layer → LSTM (with optional bidirectionality)
- **Decoder**: Embedding layer → LSTM → Linear layer → Vocabulary logits

##  Requirements

Install the required dependencies:

```bash
pip install torch datasets tokenizers huggingface_hub matplotlib
```

##  Dataset

The model uses the [BillSum dataset](https://huggingface.co/datasets/billsum), loaded through Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset('billsum')
```

##  Project Structure

```
summarization-lstm/
├── summarization-lstm.ipynb    # Main training and evaluation notebook
├── README.md                   # Project documentation
```

##  Training

Train the model by running the training loop defined in the notebook. Example:

```python
losses, accuracies = training_loop(
    model=summarizer,
    dataloader=train_loader,
    optimizer=optimizer,
    criterion=criterion,
    epochs=10,
    device=device,
    label_pad_idx=label_pad_idx
)
```

Training outputs:
- Loss per epoch
- Accuracy per epoch
- TQDM progress bar for real-time updates
> ⚠️ **Note:** The model is trained for only 10 epochs with a small architecture. It is not expected to achieve high summarization quality and is intended purely as a **showcase** of implementing sequence-to-sequence summarization with LSTMs. Also, token-level **accuracy** is not a good evaluation metric for text summarization, as it doesn't capture semantic correctness or fluency.


##  Evaluation

To evaluate the model and visualize example outputs, run:

```python
inference_loop(model=summarizer, data_loader=test_loader, tokenizer=tokenizer, device=device)
```

You can modify `max_examples` to control how many summaries are generated.

##  Configuration

Model hyperparameters can be adjusted:

- `embedding_dim`: Size of word embeddings
- `hidden_dim`: LSTM hidden size
- `num_layers`: Number of LSTM layers
- `bidirectional`: Use bidirectional LSTM (True/False)
- `dropout`: Dropout probability

##  Notes

- The tokenizer uses a shared vocabulary for input and output.
- Special tokens (`<SOS>`, `<EOS>`, `[PAD]`, `[UNK]`) are added and managed explicitly.
- Decoding is performed step by step using greedy decoding (no teacher forcing).
## Result
![plot](https://github.com/HeshamEL-Shreif/Summarization-with-LSTM/blob/main/output.png)
