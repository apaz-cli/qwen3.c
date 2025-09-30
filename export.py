#!/usr/bin/env python3

import argparse
import gzip
import json
import math
import os
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import ModelArgs, Transformer

import json
from jinja2 import Template


def serialize_fp32(file, tensor):
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f"{len(d)}f", *d)
    file.write(b)


def serialize_int8(file, tensor):
    """writes one int8 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f"{len(d)}b", *d)
    file.write(b)


def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    assert w.shape[-1] % group_size == 0

    # convert to float32 and flatten to group size
    w = w.float().reshape(-1, group_size)

    # Calculate scaling factor and scale to range [-127, 127]
    wmax = torch.abs(w).max(dim=1).values
    scaling_factor = wmax / 127.0
    quant = w / scaling_factor[:, None]
    int8val = torch.round(quant).to(torch.int8)

    # Dequantize so we can check for error
    fp32val = (int8val.float() * scaling_factor[:, None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    max_err = torch.abs(fp32valr - w).max(dim=1).values.max().item()

    return int8val, scaling_factor, max_err


def model_export(model, filepath, group_size=64):
    """
    Export the model weights in Q8_0 into .bin file to be read from C.
    That is:
    - quantize all weights to symmetric int8, in range [-127, 127]
    - all other tensors (the rmsnorm params) are kept and exported in fp32
    - quantization is done in groups of group_size to reduce the effects of any outliers
    """
    version = 1

    # let's first do some validation for this export type
    while model.params.dim % group_size != 0:
        group_size //= 2
        print(f"BACKOFF: reducing group size to {group_size} to fit hidden_dim")
    # Collect weights for validation but export sequentially by layer
    all_weights = [
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    has_shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)

    if not has_shared_classifier:
        all_weights.append(model.output.weight)
    for i, embeddings_weight in enumerate(all_weights):
        assert (
            embeddings_weight.numel() % group_size == 0
        ), f"weight {i} has numel {embeddings_weight.numel()}, not a multiple of group_size {group_size}"

    # write
    out_file = open(filepath, "wb")
    # first write out the header. the header will be 4096 bytes
    # write magic, which will be uint32 of "qwen" in ASCII
    out_file.write(struct.pack("I", 0x6E657771))
    # write the params
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack(
        "iiiiiiiiii",
        p.dim,
        hidden_dim,
        p.n_layers,
        p.n_heads,
        n_kv_heads,
        p.vocab_size,
        p.max_seq_len,
        p.head_dim,
        int(has_shared_classifier),
        group_size,
    )
    out_file.write(header)

    pad = 4096 - out_file.tell()  # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b"\0" * pad)
    # now that the header is done, let's write out the model

    # Export weights sequentially by layer for better memory locality
    ew = []
    weight_count = 0

    # First, export token embeddings
    embeddings_weight = model.tok_embeddings.weight
    q, s, err = quantize_q80(embeddings_weight, group_size)
    serialize_int8(out_file, q)
    serialize_fp32(out_file, s)
    ew.append((err, embeddings_weight.shape))
    weight_count += 1
    print(f"{weight_count}/{len(all_weights)} quantized token_embeddings {tuple(embeddings_weight.shape)} to Q8_0 with max error {err:.8f}")

    # Then, export each layer with all its data together (norms + weights)
    for layer_idx, layer in enumerate(model.layers):
        assert layer.attention.lq.weight is not None
        assert layer.attention.lk.weight is not None

        layer_weights = [
            ("rms_att", layer.attention_norm.weight, False),  # (name, weight, quantize)
            ("rms_ffn", layer.ffn_norm.weight, False),
            ("q_ln", layer.attention.lq.weight, False),
            ("k_ln", layer.attention.lk.weight, False),
            ("wq", layer.attention.wq.weight, True),
            ("wk", layer.attention.wk.weight, True),
            ("wv", layer.attention.wv.weight, True),
            ("wo", layer.attention.wo.weight, True),
            ("w1", layer.feed_forward.w1.weight, True),
            ("w2", layer.feed_forward.w2.weight, True),
            ("w3", layer.feed_forward.w3.weight, True),
        ]

        for weight_name, weight, quantize in layer_weights:
            if quantize:
                q, s, err = quantize_q80(weight, group_size)
                serialize_int8(out_file, q)
                serialize_fp32(out_file, s)
                ew.append((err, weight.shape))
                weight_count += 1
                print(f"{weight_count}/{len(all_weights)} wrote layer_{layer_idx}.{weight_name} {tuple(weight.shape)} as Q8_0 with max error {err:.8f}")
            else:
                serialize_fp32(out_file, weight)
                print(f"{weight_count+1}/{len(all_weights)} wrote layer_{layer_idx}.{weight_name} {tuple(weight.shape)} as fp32")

    # Write final norm after all layers
    serialize_fp32(out_file, model.norm.weight)

    # Finally, export classifier weights if not shared
    if not has_shared_classifier:
        embeddings_weight = model.output.weight
        q, s, err = quantize_q80(embeddings_weight, group_size)
        serialize_int8(out_file, q)
        serialize_fp32(out_file, s)
        ew.append((err, embeddings_weight.shape))
        weight_count += 1
        print(f"{weight_count}/{len(all_weights)} quantized output {tuple(embeddings_weight.shape)} to Q8_0 with max error {err:.8f}")
    else:
        print("output layer shares weights with token embeddings, not writing.")

    # print the highest error across all weights, should be very small, e.g. O(~0.001)
    ew.sort(reverse=True)
    print(f"max quantization group error across all weights: {ew[0][0]:.8f}")

    # write to binary file
    out_file.close()
    print(f"Wrote model checkpoint to {filepath}")


## Tokenizer functions


def bytes_to_unicode():
    """Reference byte -> Unicode map."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def internal_to_bytes(U2B, token_str: str) -> bytes:
    return b"".join(
        bytes([U2B[ch]]) if ch in U2B else ch.encode("utf-8") for ch in token_str
    )


def build_tokenizer(model, file):
    # Build the reverse table once
    B2U = bytes_to_unicode()
    U2B = {u: b for b, u in B2U.items()}

    # Load tokenizer (adjust as needed)
    tokenizer = model.tokenizer

    # Get ID → token mapping
    vocab = tokenizer.get_vocab()
    id_to_token = {v: k for k, v in vocab.items()}
    all_tokens = [id_to_token[i] for i in sorted(id_to_token)]

    tokenizer_path = Path(tokenizer.name_or_path)
    tokenizer_json_path = tokenizer_path / "tokenizer.json"

    with open(tokenizer_json_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    # Extract vocab and merge rules
    vocab = tokenizer_data["model"]["vocab"]
    merges = tokenizer_data["model"]["merges"]

    # Build merge rank table
    merge_rank = {
        "".join(tuple(merge if isinstance(merge, list) else merge.split())): i
        for i, merge in enumerate(merges)
    }

    # Create pseudo-score dictionary
    # Tokens from initial vocab get score 0 (unmerged tokens)
    # Merged tokens get scores based on merge rank
    pseudo_scores = {}
    for token_id, token in enumerate(all_tokens):
        # If this token was the result of a merge, it will appear in merge_rank
        rank = merge_rank.get(token)

        if rank is not None:
            score = -math.log(rank + 1)
        else:
            score = -1e6  # Initial vocab tokens
        pseudo_scores[token] = score

    # Convert all tokens to bytes
    token_bytes_list = [internal_to_bytes(U2B, t) for t in all_tokens]
    max_token_length = max(len(tb) for tb in token_bytes_list)

    # Write to binary with mmap-friendly fixed-size layout
    with open(file + ".tokenizer", "wb") as out_f:
        # Header: vocab_size, max_token_length, bos_token_id, eos_token_id
        out_f.write(struct.pack("=I", len(all_tokens)))
        out_f.write(struct.pack("=I", max_token_length))
        out_f.write(struct.pack("=I", model.bos_token_id))
        out_f.write(struct.pack("=I", model.eos_token_id))

        # Write all merge scores as contiguous array
        for token in all_tokens:
            out_f.write(struct.pack("f", pseudo_scores[token]))

        # Write all token strings with fixed-size slots
        for token_bytes in token_bytes_list:
            # Write actual token bytes
            out_f.write(token_bytes)
            # Pad with zeros to max_token_length
            padding = max_token_length - len(token_bytes)
            out_f.write(b'\0' * padding)

    print(f"Wrote tokenizer model to {file}.tokenizer")


def build_prompts(model, file):
    # Render the templates and write out the prompts
    prompts_file = "prompts.h"
    template = Template(model.tokenizer.chat_template)
    with open(prompts_file, "w", encoding="utf-8", newline="") as f:
        f.write("// This file is auto-generated by export.py\n")
        f.write("// Do not edit this file directly, edit export.py instead\n")

        messages = [{"role": "user", "content": "%s"}]
        rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
        f.write(f"const char *default_prompt_template         = \"{repr(rendered_prompt)[1:-1]}\";\n")
        rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=True)
        f.write(f"const char *thinking_prompt_template        = \"{repr(rendered_prompt)[1:-1]}\";\n")

        messages = [{"role": "system", "content": "%s"}, {"role": "user", "content": "%s"}]
        rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
        f.write(f"const char *system_prompt_template          = \"{repr(rendered_prompt)[1:-1]}\";\n")
        rendered_prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=True)
        f.write(f"const char *system_thinking_prompt_template = \"{repr(rendered_prompt)[1:-1]}\";\n")

    print(f"Wrote prompt templates to {prompts_file}.")


# -----------------------------------------------------------------------------
# Load / import functions


def load_hf_model(model_path):

    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    # load HF model
    hf_model = AutoModelForCausalLM.from_pretrained(model_path)
    hf_dict = hf_model.state_dict()

    # convert config to ModelArgs


    with open(os.path.join(model_path, "config.json"), "r") as f:
        config_json = json.load(f)
    print(json.dumps(config_json, indent=2))

    config = ModelArgs(
        dim=config_json["hidden_size"],
        n_layers=config_json["num_hidden_layers"],
        n_heads=config_json["num_attention_heads"],
        n_kv_heads=config_json["num_key_value_heads"],
        vocab_size=config_json["vocab_size"],
        hidden_dim=config_json["intermediate_size"],
        norm_eps=config_json["rms_norm_eps"],
        max_seq_len=config_json["max_position_embeddings"],
        head_dim=config_json["head_dim"],
    )

    # create a new Transformer object and set weights
    model = Transformer(config)

    model.tok_embeddings.weight = nn.Parameter(hf_dict["model.embed_tokens.weight"])
    model.norm.weight = nn.Parameter(hf_dict["model.norm.weight"])

    model.tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.bos_token_id = config_json.get("bos_token_id", 0)
    model.eos_token_id = config_json.get("eos_token_id", 0)

    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(  # type: ignore
            hf_dict[f"model.layers.{i}.input_layernorm.weight"]
        )
        layer.attention.wq.weight = nn.Parameter(  # type: ignore
            hf_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
        )
        layer.attention.wk.weight = nn.Parameter(  # type: ignore
            hf_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
        )
        layer.attention.wv.weight = nn.Parameter(  # type: ignore
            hf_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
        )
        layer.attention.wo.weight = nn.Parameter(  # type: ignore
            hf_dict[f"model.layers.{i}.self_attn.o_proj.weight"]
        )
        layer.attention.lq.weight = nn.Parameter(  # type: ignore
            hf_dict[f"model.layers.{i}.self_attn.q_norm.weight"]
        )
        layer.attention.lk.weight = nn.Parameter(  # type: ignore
            hf_dict[f"model.layers.{i}.self_attn.k_norm.weight"]
        )
        layer.ffn_norm.weight = nn.Parameter(  # type: ignore
            hf_dict[f"model.layers.{i}.post_attention_layernorm.weight"]
        )
        layer.feed_forward.w1.weight = nn.Parameter(  # type: ignore
            hf_dict[f"model.layers.{i}.mlp.gate_proj.weight"]
        )
        layer.feed_forward.w2.weight = nn.Parameter(  # type: ignore
            hf_dict[f"model.layers.{i}.mlp.down_proj.weight"]
        )
        layer.feed_forward.w3.weight = nn.Parameter(  # type: ignore
            hf_dict[f"model.layers.{i}.mlp.up_proj.weight"]
        )

    # final classifier
    model.output.weight = nn.Parameter(hf_dict["lm_head.weight"])
    model.eval()
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("hfpath", type=str, help="huggingface model path")
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument("promptpath", type=str, help="the output prompts filepath")
    args = parser.parse_args()

    if args.filepath is None or args.hfpath is None:
        print("Usage: export.py <huggingface_input_path> <output.bin> <output_prompts.h>")
        print("")
        print("e.g.   git clone https://huggingface.co/Qwen/Qwen3-4B")
        print("       export.py Qwen/Qwen3-4B Qwen3-4B.bin prompts.h")
        print("")
        exit(0)

    model = load_hf_model(args.hfpath)

    # export
    build_tokenizer(model, args.filepath)
    build_prompts(model, args.promptpath)
    model_export(model, args.filepath)
