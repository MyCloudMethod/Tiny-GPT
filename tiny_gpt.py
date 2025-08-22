#!/usr/bin/env python3

import argparse
import dataclasses
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# --------------- Reproducibility ---------------

def seed_all(seed: int = 1337) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # can slow down, not needed on CPU


# --------------- Data utilities ---------------

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_tiny_shakespeare(data_dir: str) -> str:
    """
    Downloads tiny Shakespeare to data_dir/input.txt if not present.
    Returns file path.
    """
    ensure_dir(data_dir)
    dst = os.path.join(data_dir, "input.txt")
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst

    try:
        import urllib.request

        print(f"Downloading Tiny Shakespeare to {dst} ...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, dst)
        print("Download complete.")
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}")

    return dst


class CharTokenizer:
    """
    Simple character-level tokenizer.
    Builds vocabulary from a text corpus and provides encode/decode.
    """
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: List[str] = chars
        self.vocab_size = len(chars)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def to_meta(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "itos": self.itos,
        }

    @staticmethod
    def from_meta(meta: dict) -> "CharTokenizer":
        tok = object.__new__(CharTokenizer)
        tok.itos = meta["itos"]
        tok.stoi = {ch: i for i, ch in enumerate(tok.itos)}
        tok.vocab_size = len(tok.itos)
        return tok


def load_text_dataset(data_dir: str) -> str:
    path = download_tiny_shakespeare(data_dir)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_train_val(text: str, split: float = 0.9) -> Tuple[str, str]:
    n = len(text)
    split_ix = int(n * split)
    return text[:split_ix], text[split_ix:]


class CharBlockDataset(Dataset):
    """
    Produces (x, y) where:
      x: block_size tokens
      y: block_size tokens, shifted by one (next-token prediction)
    """
    def __init__(self, data_ids: torch.Tensor, block_size: int):
        super().__init__()
        self.data = data_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


# --------------- Model ---------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # causal mask to ensure tokens only attend to previous tokens
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)   # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))       # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )


def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
    
    
    class Block(nn.Module):
        def __init__(self, n_embd: int, n_head: int, dropout: float, block_size: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(n_embd)
            self.attn = CausalSelfAttention(n_embd, n_head, dropout, block_size)
            self.ln2 = nn.LayerNorm(n_embd)
            self.mlp = MLP(n_embd, dropout)
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x
    
    
    class TinyGPT(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            block_size: int,
            n_layer: int = 4,
            n_head: int = 4,
            n_embd: int = 128,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.vocab_size = vocab_size
            self.block_size = block_size
    
            self.tok_emb = nn.Embedding(vocab_size, n_embd)
            self.pos_emb = nn.Embedding(block_size, n_embd)
            self.drop = nn.Dropout(dropout)
            self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, vocab_size, bias=False)
    
            self.apply(self._init_weights)
    
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
        def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
            B, T = idx.size()
            if T > self.block_size:
                idx = idx[:, -self.block_size:]
                T = idx.size(1)
    
            tok = self.tok_emb(idx)  # (B, T, C)
            pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T, C)
            x = tok + pos.unsqueeze(0)  # (B, T, C)
            x = self.drop(x)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            logits = self.head(x)  # (B, T, vocab_size)
    
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
            return logits, loss
    
        @torch.no_grad()
        def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
            self.eval()
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]
                logits, _ = self(idx_cond)  # (B, T, V)
                logits = logits[:, -1, :] / max(temperature, 1e-5)
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)
            return idx
    
    
    @dataclass
    class TrainConfig:
        data_dir: str = "./data"
        out_dir: str = "./out"
        max_steps: int = 200
        batch_size: int = 32
        block_size: int = 128
        n_layer: int = 4
        n_head: int = 4
        n_embd: int = 192
        dropout: float = 0.1
        lr: float = 3e-4
        eval_interval: int = 50
        eval_iters: int = 50
        grad_clip: float = 1.0
        seed: int = 1337
        device: str = "auto"
        checkpoint: str = ""
    
    
    def get_device(pref: str = "auto") -> torch.device:
        if pref != "auto":
            return torch.device(pref)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    
    def get_batch(data_ids: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
        ix = torch.randint(0, len(data_ids) - block_size - 1, (batch_size,))
        x = torch.stack([data_ids[i : i + block_size] for i in ix])
        y = torch.stack([data_ids[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)
    
    
    @torch.no_grad()
    def estimate_loss(
        model: nn.Module,
        train_ids: torch.Tensor,
        val_ids: torch.Tensor,
        block_size: int,
        batch_size: int,
        eval_iters: int,
        device: torch.device,
    ) -> Dict[str, float]:
        model.eval()
        out = {}
        for split, data in [("train", train_ids), ("val", val_ids)]:
            losses = []
            for _ in range(eval_iters):
                xb, yb = get_batch(data, block_size, batch_size, device)
                _, loss = model(xb, yb)
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out
    
    
    def count_params(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())
    
    
    def save_ckpt(path: str, model: TinyGPT, optimizer: torch.optim.Optimizer, step: int, cfg: TrainConfig, tokenizer: CharTokenizer):
        ensure_dir(os.path.dirname(path) or ".")
        payload = {
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict(),
            "step": step,
            "config": dataclasses.asdict(cfg),
            "model_hparams": {
                "vocab_size": model.vocab_size,
                "block_size": model.block_size,
            },
            "tokenizer_meta": tokenizer.to_meta(),
        }
        torch.save(payload, path)
    
    
    def load_ckpt(path: str, device: torch.device):
        payload = torch.load(path, map_location=device)
        return payload
    
    
    def action_train(args):
        seed_all(args.seed)
        device = get_device(args.device)
        print(f"Using device: {device}")
    
        text = load_text_dataset(args.data_dir)
        tokenizer = CharTokenizer(text)
        ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
        train_text, val_text = split_train_val(text, split=0.9)
        train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
        val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    
        cfg = TrainConfig(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
            lr=args.lr,
            eval_interval=args.eval_interval,
            eval_iters=args.eval_iters,
            grad_clip=args.grad_clip,
            seed=args.seed,
            device=str(device),
            checkpoint=args.checkpoint or os.path.join(args.out_dir, "ckpt.pt"),
        )
    
        model = TinyGPT(
            vocab_size=tokenizer.vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
            dropout=cfg.dropout,
        ).to(device)
    
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.01)
    
        start_step = 0
        if cfg.checkpoint and os.path.exists(cfg.checkpoint):
            print(f"Loading checkpoint from {cfg.checkpoint}")
            payload = load_ckpt(cfg.checkpoint, device)
            try:
                model.load_state_dict(payload["model_state"])
                optimizer.load_state_dict(payload["opt_state"])
                start_step = int(payload.get("step", 0))
                print(f"Resumed at step {start_step}")
            except Exception as e:
                print(f"Warning: failed to load checkpoint: {e}")
    
        print(f"Model parameters: {count_params(model)/1e6:.2f}M")
        t0 = time.time()
        for step in range(start_step, cfg.max_steps):
            xb, yb = get_batch(train_ids, cfg.block_size, cfg.batch_size, device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
    
            if (step % cfg.eval_interval == 0) or (step == cfg.max_steps - 1):
                losses = estimate_loss(model, train_ids, val_ids, cfg.block_size, cfg.batch_size, cfg.eval_iters, device)
                dt = time.time() - t0
                t0 = time.time()
                print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | last {loss.item():.4f} | {dt:.2f}s")
                ckpt_path = cfg.checkpoint
                try:
                    save_ckpt(ckpt_path, model, optimizer, step, cfg, tokenizer)
                    print(f"Saved checkpoint to {ckpt_path}")
                except Exception as e:
                    print(f"Failed to save checkpoint: {e}")
    
        # Quick sample for sanity
        prompt = "ROMEO:"
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=50)
        print(tokenizer.decode(out[0].tolist()))
    
    
    def action_sample(args):
        device = get_device(args.device)
        ckpt = args.checkpoint or os.path.join(args.out_dir, "ckpt.pt")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        payload = load_ckpt(ckpt, device)
        tokenizer = CharTokenizer.from_meta(payload["tokenizer_meta"])
        mh = payload.get("model_hparams", {})
        vocab_size = mh.get("vocab_size", tokenizer.vocab_size)
        block_size = mh.get("block_size", args.block_size)
    
        model = TinyGPT(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        ).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
    
        prompt = args.prompt or ""
        if prompt == "":
            prompt = "ROMEO:"
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(context, max_new_tokens=args.num_tokens, temperature=args.temperature, top_k=args.top_k)
        text = tokenizer.decode(out[0].tolist())
        print(text)
    
    
    def build_arg_parser():
        p = argparse.ArgumentParser(description="Tiny GPT from scratch (char-level)")
        p.add_argument("--action", choices=["train", "sample"], default="train")
        p.add_argument("--data-dir", type=str, default="./data")
        p.add_argument("--out-dir", type=str, default="./out")
        p.add_argument("--checkpoint", type=str, default="")
        p.add_argument("--max-steps", type=int, default=200)
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--block-size", type=int, default=128)
        p.add_argument("--n-layer", type=int, default=4)
        p.add_argument("--n-head", type=int, default=4)
        p.add_argument("--n-embd", type=int, default=192)
        p.add_argument("--dropout", type=float, default=0.1)
        p.add_argument("--lr", type=float, default=3e-4)
        p.add_argument("--eval-interval", type=int, default=50)
        p.add_argument("--eval-iters", type=int, default=50)
        p.add_argument("--grad-clip", type=float, default=1.0)
        p.add_argument("--seed", type=int, default=1337)
        p.add_argument("--device", type=str, default="auto")
        # sampling
        p.add_argument("--prompt", type=str, default="")
        p.add_argument("--num-tokens", type=int, default=400)
        p.add_argument("--temperature", type=float, default=1.0)
        p.add_argument("--top-k", type=int, default=50)
        return p
    
    
    def main():
        args = build_arg_parser().parse_args()
        if args.action == "train":
            action_train(args)
        elif args.action == "sample":
            action_sample(args)
        else:
            raise ValueError(f"Unknown action {args.action}")
    
    
    if __name__ == "__main__":
        main()
