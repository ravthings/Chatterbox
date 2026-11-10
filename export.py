import json, os
import mlx.core as mx
from mlx.utils import tree_flatten
from model import TinyModel, n_embd, vocab_size, block_size

EOT = 50256
model = TinyModel()
model.load_weights("chatterbox.safetensors")
w = dict(tree_flatten(model.parameters()))

hf = {
    "model.embed_tokens.weight": w["tok_emb.weight"],
    "model.norm.weight": w["ln_f.weight"],
}
for i in range(4):
    p, h = f"blocks.layers.{i}.", f"model.layers.{i}."
    hf[h + "input_layernorm.weight"] = w[p + "ln1.weight"]
    hf[h + "post_attention_layernorm.weight"] = w[p + "ln2.weight"]
    for ours, theirs in [("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj")]:
        hf[h + f"self_attn.{theirs}.weight"] = mx.concatenate(
            [w[p + f"sa.heads.{j}.{ours}.weight"] for j in range(4)], axis=0)
    hf[h + "self_attn.o_proj.weight"] = w[p + "sa.proj.weight"]
    for name in ["gate_proj", "up_proj", "down_proj"]:
        hf[h + f"mlp.{name}.weight"] = w[p + f"ffwd.{name}.weight"]

os.makedirs("export", exist_ok=True)
mx.save_safetensors("export/model.safetensors", hf, metadata={"format": "pt"})

config = {
    "architectures": ["LlamaForCausalLM"], "model_type": "llama",
    "hidden_size": n_embd, "intermediate_size": 4 * n_embd * 2 // 3,
    "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 4,
    "head_dim": n_embd // 4, "vocab_size": vocab_size,
    "max_position_embeddings": block_size, "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0, "tie_word_embeddings": True, "hidden_act": "silu",
    "bos_token_id": EOT, "eos_token_id": EOT, "torch_dtype": "float32",
}
with open("export/config.json", "w") as f:
    json.dump(config, f, indent=2)
print("exported", len(hf), "tensors to export/")