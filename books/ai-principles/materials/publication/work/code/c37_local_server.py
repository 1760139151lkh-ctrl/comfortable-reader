"""Small localhost-only C19 model service with explicit equal-length batch endpoint."""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from c37_cached_decode import BOS, EOS, PAD, decode_one, generate, load_serving, prefill, select_token


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate_equal_length_batch(model, windows, max_new):
    if len({len(x) for x in windows}) != 1:
        raise ValueError("all batch prompts need the same tokenized window length")
    device = next(model.parameters()).device
    tokens = torch.tensor(windows, dtype=torch.long, device=device)
    logits, cache = prefill(model, tokens)
    generated = [[] for _ in windows]
    finished = [False] * len(windows)
    rebuilds = 0
    for step in range(max_new):
        chosen = select_token(logits, method="greedy")
        for i, token in enumerate(chosen.tolist()):
            if not finished[i]:
                generated[i].append(token)
                if token == EOS:
                    finished[i] = True
        if all(finished) or step + 1 == max_new:
            break
        # Ended items remain as EOS placeholders until the rest finish; their
        # later logits are ignored. Active rows retain equal lengths.
        chosen = chosen.clone()
        for i, done in enumerate(finished):
            if done:
                chosen[i] = EOS
        tokens = torch.cat((tokens, chosen[:, None]), dim=1)
        if cache[0][0].shape[2] == model.cfg.context:
            logits, cache = prefill(model, tokens[:, -model.cfg.context:])
            rebuilds += 1
        else:
            logits, cache = decode_one(model, chosen, cache)
    return generated, rebuilds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()
    if not 0 <= a.port <= 65535:
        raise ValueError("port outside 0..65535")
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    torch.set_num_threads(8)
    model, tokenizer, state = load_serving(device)
    lock = threading.Lock()  # independent HTTP handler threads do not overlap GPU model use

    def encode_prompt(prompt):
        if not isinstance(prompt, str) or not prompt or len(prompt) > 4000:
            raise ValueError("prompt must be a nonempty string of at most 4000 characters")
        ordinary = tokenizer.encode(prompt, add_special_tokens=False).ids
        if any(token in (PAD, BOS, EOS) for token in ordinary):
            raise ValueError("special control tokens are not accepted inside the user prompt")
        original = [BOS] + ordinary
        window = original[-model.cfg.context:]
        return window, len(original) - len(window)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # No prompt text in HTTP logs.
            return

        def respond(self, status, item):
            data = (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health":
                self.respond(404, {"error": "unknown path"})
                return
            self.respond(200, {"ready": True, "device": str(device), "model_source_sha256": state["source_checkpoint_sha256"], "model_training_epoch": state["selected_epoch"], "context_positions": model.cfg.context, "batching": "explicit equal-token-length batches only; inference lock serializes unrelated requests"})

        def do_POST(self):
            if self.path not in ("/generate", "/generate_batch"):
                self.respond(404, {"error": "unknown path"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("JSON request byte length must be 1..65536")
                payload = json.loads(self.rfile.read(length))
                max_new = payload.get("max_new_tokens", 16)
                if not isinstance(max_new, int) or not 1 <= max_new <= 64:
                    raise ValueError("max_new_tokens must be an integer 1..64")
                if self.path == "/generate":
                    window, dropped = encode_prompt(payload.get("prompt"))
                    method = payload.get("selection", "greedy")
                    if method not in ("greedy", "sample_top40"):
                        raise ValueError("selection must be greedy or sample_top40")
                    temperature = payload.get("temperature", 0.8)
                    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0.1 <= temperature <= 5.0:
                        raise ValueError("temperature must be a finite number within 0.1..5.0")
                    if method == "greedy" and "temperature" in payload:
                        raise ValueError("temperature applies only to sample_top40")
                    seed = payload.get("seed", 20260925)
                    if not isinstance(seed, int):
                        raise ValueError("seed must be an integer")
                    with lock:
                        sync(device)
                        start = time.perf_counter()
                        result = generate(model, window, max_new, "cached", method=method, seed=seed, temperature=temperature)
                        sync(device)
                        elapsed = (time.perf_counter() - start) * 1000
                    text = tokenizer.decode([i for i in result["new_token_ids"] if i != EOS], skip_special_tokens=False)
                    self.respond(200, {"text": text, "new_token_ids": result["new_token_ids"], "ended_with_eos": result["ended_with_eos"], "input_tokens_outside_128_window": dropped, "selection": method, "temperature": temperature if method == "sample_top40" else None, "context_rebuilds": result["context_rebuilds_after_window_overflow"], "elapsed_model_ms": elapsed, "identity": "fixed C19 trained model, no update"})
                else:
                    prompts = payload.get("prompts")
                    if not isinstance(prompts, list) or not 1 <= len(prompts) <= 8:
                        raise ValueError("prompts must be a list of 1..8 strings")
                    encoded = [encode_prompt(prompt) for prompt in prompts]
                    windows = [item[0] for item in encoded]
                    if len({len(x) for x in windows}) != 1:
                        raise ValueError("this batch route needs equal tokenized prompt lengths")
                    with lock:
                        sync(device)
                        start = time.perf_counter()
                        generations, rebuilds = generate_equal_length_batch(model, windows, max_new)
                        sync(device)
                        elapsed = (time.perf_counter() - start) * 1000
                    self.respond(200, {"outputs": [{"text": tokenizer.decode([i for i in ids if i != EOS], skip_special_tokens=False), "new_token_ids": ids, "ended_with_eos": bool(ids and ids[-1] == EOS), "input_tokens_outside_128_window": encoded[i][1]} for i, ids in enumerate(generations)], "selection": "greedy", "context_rebuilds": rebuilds, "elapsed_model_ms": elapsed, "identity": "one explicit same-length batch, fixed C19 trained model"})
            except (ValueError, json.JSONDecodeError) as error:
                self.respond(400, {"error": str(error)})
            except Exception as error:
                self.respond(500, {"error": type(error).__name__, "detail": str(error)})

    server = ThreadingHTTPServer((a.host, a.port), Handler)
    server.daemon_threads = True
    print(json.dumps({"host": a.host, "port": server.server_address[1], "pid": os.getpid(), "device": str(device), "model_source_sha256": state["source_checkpoint_sha256"], "context": model.cfg.context, "note": "localhost-only; Ctrl-C stops server"}), flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
