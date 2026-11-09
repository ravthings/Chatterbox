from datasets import load_dataset
import tiktoken, json

enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token
block_size = 128

ds = load_dataset("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft")