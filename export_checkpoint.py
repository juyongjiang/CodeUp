import os
import json
import fire
import torch
import transformers
from typing import List
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402

def export_checkpoint(
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    export_dir: str = "export_checkpoint",
    checkpoint_type: str = "hf",
):
    assert (
        checkpoint_type
    ), "Please specify a --checkpoint_type, e.g. --checkpoint_type='hf'"

    BASE_MODEL = base_model or os.environ.get("BASE_MODEL", None)
    assert (
        BASE_MODEL
    ), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=huggyllama/llama-7b`"  # noqa: E501

    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_weights,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

    if checkpoint_type.lower() == "hf":
        # merge weights - new merging method from peft
        lora_model = lora_model.merge_and_unload()

        lora_model.train(False)

        lora_model_sd = lora_model.state_dict()
        deloreanized_sd = {
            k.replace("base_model.model.", ""): v
            for k, v in lora_model_sd.items()
            if "lora" not in k
        }

        LlamaForCausalLM.save_pretrained(
            base_model, f"{export_dir}/hf_ckpt", state_dict=deloreanized_sd, max_shard_size="400MB"
        )
    elif checkpoint_type.lower() == "pytorch":
        # merge weights
        for layer in lora_model.base_model.model.model.layers:
            if 'q_proj' in lora_target_modules:
                layer.self_attn.q_proj.merge_weights = True
            if 'k_proj' in lora_target_modules:
                layer.self_attn.k_proj.merge_weights = True
            if 'v_proj' in lora_target_modules:
                layer.self_attn.v_proj.merge_weights = True
            if 'o_proj' in lora_target_modules:
                layer.self_attn.o_proj.merge_weights = True

        lora_model.train(False)

        lora_model_sd = lora_model.state_dict()
        new_state_dict = {}
        for k, v in lora_model_sd.items():
            new_k = translate_state_dict_key(k)
            if new_k is not None:
                if "wq" in new_k or "wk" in new_k:
                    new_state_dict[new_k] = unpermute(v)
                else:
                    new_state_dict[new_k] = v

        os.makedirs(f"{export_dir}/ckpt", exist_ok=True)
        torch.save(new_state_dict, os.path.join(export_dir, "ckpt/consolidated.00.pth"))

        # record params of Llama
        params = {
            "dim": 4096,
            "multiple_of": 256,
            "n_heads": 32,
            "n_layers": 32,
            "norm_eps": 1e-06,
            "vocab_size": -1,
        }
        n_layers = params["n_layers"]
        n_heads = params["n_heads"]
        dim = params["dim"]
        dims_per_head = dim // n_heads
        base = 10000.0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head)
        )
        with open(os.path.join(export_dir, "ckpt/params.json"), "w") as f:
            json.dump(params, f)
    else:
        print(checkpoint_type)
        raise NotImplementedError


def translate_state_dict_key(k):  # noqa: C901
    k = k.replace("base_model.model.", "")
    if k == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    elif k == "model.norm.weight":
        return "norm.weight"
    elif k == "lm_head.weight":
        return "output.weight"
    elif k.startswith("model.layers."):
        layer = k.split(".")[2]
        if k.endswith(".self_attn.q_proj.weight"):
            return f"layers.{layer}.attention.wq.weight"
        elif k.endswith(".self_attn.k_proj.weight"):
            return f"layers.{layer}.attention.wk.weight"
        elif k.endswith(".self_attn.v_proj.weight"):
            return f"layers.{layer}.attention.wv.weight"
        elif k.endswith(".self_attn.o_proj.weight"):
            return f"layers.{layer}.attention.wo.weight"
        elif k.endswith(".mlp.gate_proj.weight"):
            return f"layers.{layer}.feed_forward.w1.weight"
        elif k.endswith(".mlp.down_proj.weight"):
            return f"layers.{layer}.feed_forward.w2.weight"
        elif k.endswith(".mlp.up_proj.weight"):
            return f"layers.{layer}.feed_forward.w3.weight"
        elif k.endswith(".input_layernorm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif k.endswith(".post_attention_layernorm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
            return None
        else:
            print(layer, k)
            raise NotImplementedError
    else:
        print(k)
        raise NotImplementedError

def permute(w):
    return (
        w.view(n_heads, dim // n_heads // 2, 2, dim)
        .transpose(1, 2)
        .reshape(dim, dim)
    )

def unpermute(w):
    return (
        w.view(n_heads, 2, dim // n_heads // 2, dim)
        .transpose(1, 2)
        .reshape(dim, dim)
    )


if __name__ == "__main__":
    fire.Fire(export_checkpoint)