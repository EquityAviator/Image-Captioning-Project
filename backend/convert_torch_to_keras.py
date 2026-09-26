"""
CaptionAI — Stage 4: convert PyTorch decoder weights to Keras model.h5.

Builds the exact Keras architecture (backend/model/architecture.py),
copies the torch state_dict in, and ONLY saves weights/model.h5 if a
logit-comparison gate passes: torch vs Keras softmax outputs on 50
random feature/sequence pairs must match within 1e-4.

Keras LSTM gate order is [i, f, c, o]; PyTorch is [i, f, g, o] — the same,
so no permutation is needed (the gate verifies this empirically).

Usage:
    python convert_torch_to_keras.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
WEIGHTS_DIR = PROJECT_DIR / "weights"

sys.path.insert(0, str(BACKEND_DIR))
from model.architecture import build_decoder  # noqa: E402

TORCH_LAYER_ORDER = ["img_fc", "embedding", "lstm", "fc128", "out"]
ATOL = 1e-4


def convert(torch_path: Path, vocab_size: int, max_length: int):
    print(">> loading torch checkpoint …", flush=True)
    ckpt = torch.load(torch_path, map_location="cpu", weights_only=True)
    sd = ckpt["state_dict"]

    print(">> building Keras decoder …", flush=True)
    keras_model = build_decoder(vocab_size, max_length)

    # img_fc: Dense(256) — keras weights (256, 1920), bias (256,)
    keras_model.get_layer("img_fc").set_weights(
        [sd["img_fc.weight"].numpy().T, sd["img_fc.bias"].numpy()])

    # embedding: Embedding(vocab, 256) — same layout
    keras_model.get_layer("caption_embedding").set_weights(
        [sd["embedding.weight"].numpy()])

    # lstm: kernel (in, 4u), recurrent_kernel (u, 4u), bias (4u,)
    # torch: weight_ih (4u, in), weight_hh (4u, u), b_ih + b_hh (4u,)
    w_ih = sd["lstm.weight_ih_l0"].numpy()
    w_hh = sd["lstm.weight_hh_l0"].numpy()
    b_ih = sd["lstm.bias_ih_l0"].numpy()
    b_hh = sd["lstm.bias_hh_l0"].numpy()
    keras_model.get_layer("decoder_lstm").set_weights(
        [w_ih.T, w_hh.T, b_ih + b_hh])

    # fc128: Dense(128) — keras weights (128, 256)
    keras_model.get_layer("fc_128").set_weights(
        [sd["fc128.weight"].numpy().T, sd["fc128.bias"].numpy()])

    # vocab_softmax: Dense(vocab) — keras weights (vocab, 128)
    keras_model.get_layer("vocab_softmax").set_weights(
        [sd["out.weight"].numpy().T, sd["out.bias"].numpy()])

    # ---- logit verification gate ---------------------------------------
    print(">> verification gate: comparing torch vs keras outputs …", flush=True)
    torch_model = _build_torch(vocab_size, max_length, sd)
    torch_model.eval()

    rng = np.random.default_rng(0)
    max_diff = 0.0
    for _ in range(50):
        feat = rng.standard_normal((1, 1920)).astype("float32")
        seq = rng.integers(0, vocab_size, size=(1, max_length)).astype("int64")
        with torch.no_grad():
            t_logits = torch_model(
                torch.tensor(feat), torch.tensor(seq))
            t_probs = torch.softmax(t_logits, dim=-1).numpy()
        k_probs = keras_model.predict([feat, seq], verbose=0, batch_size=1)
        max_diff = max(max_diff, float(np.abs(t_probs - k_probs).max()))

    print(f">> max |torch - keras| = {max_diff:.3e}", flush=True)
    if max_diff > ATOL:
        print("!! GATE FAILED — weights NOT saved. Check LSTM gate order.", flush=True)
        return 1

    print(">> gate passed — saving weights/model.h5 …", flush=True)
    keras_model.save(WEIGHTS_DIR / "model.h5")
    print(f">> saved {WEIGHTS_DIR / 'model.h5'}", flush=True)

    meta = json.loads((WEIGHTS_DIR / "metadata.json").read_text())
    meta["conversion"] = {
        "torch_checkpoint": torch_path.name,
        "gate_max_diff": round(max_diff, 8),
        "gate_tolerance": ATOL,
    }
    (WEIGHTS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    return 0


def _build_torch(vocab_size: int, max_length: int, sd):
    """Minimal torch replica of the decoder for output comparison."""
    import torch.nn as nn

    class D(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_fc = nn.Linear(1920, 256)
            self.embedding = nn.Embedding(vocab_size, 256)
            self.lstm = nn.LSTM(256, 256, batch_first=True)
            self.dropout1 = nn.Dropout(0.5)
            self.fc128 = nn.Linear(256, 128)
            self.dropout2 = nn.Dropout(0.5)
            self.out = nn.Linear(128, vocab_size)

        def forward(self, features, tokens):
            img = torch.relu(self.img_fc(features))
            emb = self.embedding(tokens)
            merged = torch.cat([img.unsqueeze(1), emb], dim=1)
            h, _ = self.lstm(merged)
            h = h[:, -1, :]
            h = self.dropout1(h)
            h = h + img
            h = torch.relu(self.fc128(h))
            h = self.dropout2(h)
            return self.out(h)

    m = D()
    m.load_state_dict(sd)
    return m


if __name__ == "__main__":
    meta = json.loads((WEIGHTS_DIR / "metadata.json").read_text())
    sys.exit(convert(
        WEIGHTS_DIR / "decoder_torch.pt",
        int(meta["vocab_size"]),
        int(meta["max_length"]),
    ))