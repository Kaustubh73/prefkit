from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable

GenerateFn = Callable[[str, str], str]

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


def ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1)
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def backend_from_env() -> str:
    return os.environ.get("PREFKIT_BACKEND", "hf")


def make_ollama_generate(model: str, decode: dict) -> GenerateFn:
    def generate_fn(prompt: str, system: str) -> str:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": decode["temperature"],
                "top_p": decode["top_p"],
                "num_predict": decode["max_new_tokens"],
                "seed": decode["seed"],
            },
        }
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode())
        msg = payload.get("message") or {}
        return str(msg.get("content") or "")

    return generate_fn


def make_hf_generate(model_id: str, decode: dict) -> GenerateFn:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    def generate_fn(prompt: str, system: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        kw = dict(tokenize=False, add_generation_prompt=True)
        try:
            text = tok.apply_chat_template(messages, enable_thinking=False, **kw)
        except TypeError:
            text = tok.apply_chat_template(messages, **kw)
        inputs = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            do_sample=decode["do_sample"],
            temperature=decode["temperature"],
            top_p=decode["top_p"],
            max_new_tokens=decode["max_new_tokens"],
        )
        gen = out[0, inputs["input_ids"].shape[1] :]
        return tok.decode(gen, skip_special_tokens=True)

    return generate_fn


def make_generate_fn(backend: str, hf_id: str, ollama_tag: str | None, decode: dict) -> GenerateFn:
    if backend == "hf":
        return make_hf_generate(hf_id, decode)
    if backend == "ollama":
        tag = ollama_tag or hf_id
        return make_ollama_generate(tag, decode)
    raise ValueError(f"unknown backend {backend}")
