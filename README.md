# Chatterbox 💬📦

A tiny 7M-parameter chatbot, made by fine-tuning my story model ravthings/bedtime on about 6,300 simple question-and-answer pairs from everyday-conversations. Built from scratch in MLX.

Chat with it.

## To run

```
ollama run ravthings/chatterbox
```

## Demo

https://github.com/user-attachments/assets/3d680ebe-7ba9-47fe-8142-5e44892e0188


## Example

> Hi
> Hi! How can I help you?

## How it works

- **Tokenizer:** BPE tokenizer
- **Model:** a decoder-only transformer
  - token and position embeddings
  - 4 blocks, each with 4-head self-attention and a feed-forward layer
  - residual connections and LayerNorm
  - embedding size [64 / 128], context of 128 characters
  - about 1 000 000 parameters
- **Training:** AdamW (learning rate 0.001, weight decay 0.05), 5,000 steps on a Mac

## Results

| 3 | 0.96 |
|---|---|
| One attention head | 2.16 |
| 4 blocks × 4 heads | **0.95** |

Going from one attention head to a full transformer more than halved the loss. The model learned real sentences, character names and story structure.

## How to run

```
pip install mlx datasets
python get_data.py
python train.py
```

Choose `y` to train (it carries on from the saved model if there is one) or `n` to load the saved model. Then type the start of a story and watch it continue. Type `quit` to stop.

## Limitations

- It only sees 128 characters at a time, so stories wander off topic.
- Because it works letter by letter, it sometimes invents words.

## Next steps

- A BPE tokenizer, so it works with word pieces instead of letters
- A bigger model and longer context

## Credits
TinyStories by Eldan & Li.
