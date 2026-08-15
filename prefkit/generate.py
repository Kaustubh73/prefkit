from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable

GenerateFn = Callable[..., str]

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


def ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=1)
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def backend_from_env() -> str:
    return os.environ.get("PREFKIT_BACKEND", "hf")


def make_seed_iter(decode: dict):
    i = {"n": 0}

    def next_seed() -> int:
        s = int(decode["seed"]) + i["n"]
        i["n"] += 1
        return s

    return next_seed


def make_ollama_generate(model: str, decode: dict) -> GenerateFn:
    next_seed = make_seed_iter(decode)

    def generate_fn(prompt: str, system: str, allowed: tuple[str, ...] | None = None) -> str:
        del allowed
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
                "seed": next_seed(),
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


def _allowed_token_ids(tok, strings: tuple[str, ...]) -> list[int]:
    ids: set[int] = set()
    unk = getattr(tok, "unk_token_id", None)
    for s in strings:
        for variant in (s, " " + s):
            enc = tok.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids.add(int(enc[0]))
        tid = tok.convert_tokens_to_ids(s)
        if tid is not None and tid != unk:
            ids.add(int(tid))
    if not ids:
        raise ValueError(f"no token ids for allowed {strings}")
    return sorted(ids)


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
    next_seed = make_seed_iter(decode)
    allowed_cache: dict[tuple[str, ...], list[int]] = {}

    def generate_fn(prompt: str, system: str, allowed: tuple[str, ...] | None = None) -> str:
        from transformers import set_seed

        set_seed(next_seed())
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
        gen_kw = dict(
            do_sample=decode["do_sample"],
            temperature=decode["temperature"],
            top_p=decode["top_p"],
            # yaml max_new_tokens stays 32; one token when alphabet is masked
            max_new_tokens=1 if allowed else decode["max_new_tokens"],
        )
        if allowed:
            ids = allowed_cache.setdefault(allowed, _allowed_token_ids(tok, allowed))
            gen_kw["prefix_allowed_tokens_fn"] = lambda _bid, _input_ids: ids
        out = model.generate(**inputs, **gen_kw)
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
