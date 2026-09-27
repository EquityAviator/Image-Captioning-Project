"""
Calibrate the BLIP semantic-score logistic against GENERATED captions.

Decodes N validation images with the deployed model (torch twin, beam-5),
computes PMI for each generated caption, and reports the distribution so
_PMI_CENTER / _PMI_SCALE can be set to meaningful values (median / spread).

Output: experiments/confidence/semantic_calibration.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND = Path(__file__).resolve().parent
PROJECT = BACKEND.parent
WEIGHTS = PROJECT / "weights"
sys.path.insert(0, str(BACKEND))

from train_torch import CaptionDecoder  # noqa: E402
from inference import blip_scorer as bs  # noqa: E402

N_IMAGES = 150
BEAM = 5
ALPHA = 1.2

meta = json.loads((WEIGHTS / "metadata.json").read_text())
val_images = json.loads((WEIGHTS / "val_images.json").read_text())[:N_IMAGES]
image_names = json.loads((WEIGHTS / "image_names.json").read_text())
feat_index = {n: i for i, n in enumerate(image_names)}
features = np.load(WEIGHTS / "features.npy")

import torch  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CaptionDecoder(int(meta["vocab_size"]), int(meta["max_length"]))
blob = torch.load(WEIGHTS / "decoder_torch.pt", map_location="cpu", weights_only=False)
sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
model.load_state_dict(sd)
model.eval().to(device)

with open(WEIGHTS / "tokenizer.pkl", "rb") as fh:
    tok = pickle.load(fh)
rev = {v: k for k, v in tok.word_index.items()}
START = tok.word_index.get("startseq", 1)
END = tok.word_index.get("endseq", 2)
MAXL = int(meta["max_length"])


def greedy_caption(feats_row):
    tokens = [START]
    for _ in range(MAXL):
        seq = np.zeros(MAXL, dtype="int64")
        seq[: len(tokens)] = tokens
        with torch.no_grad():
            logits = model(torch.from_numpy(feats_row[None]).to(device).float(),
                           torch.from_numpy(seq[None]).to(device))
        idx = int(logits.argmax(-1))
        if idx == END:
            break
        tokens.append(idx)
    words = [rev[t] for t in tokens if t not in (START, END) and t in rev]
    return " ".join(words)


bs._load()
from PIL import Image  # noqa: E402

pmis = []
records = []
for i, img_name in enumerate(val_images):
    feat_row = features[feat_index[img_name]]
    cap = greedy_caption(feat_row)
    if not cap.strip():
        continue
    img = Image.open(PROJECT / "dataset" / "Images" / img_name).convert("RGB")
    out = bs.score(img, cap)
    if out is None:
        continue
    sem, pmi = out
    pmis.append(pmi)
    records.append({"image": img_name, "caption": cap, "pmi": pmi, "sem": sem})
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(val_images)}", flush=True)

pmis = np.array(pmis)
q = np.percentile(pmis, [10, 25, 50, 75, 90])
result = {
    "n": int(len(pmis)),
    "mean": round(float(pmis.mean()), 4),
    "p10": round(float(q[0]), 4), "p25": round(float(q[1]), 4),
    "median": round(float(q[2]), 4), "p75": round(float(q[3]), 4),
    "p90": round(float(q[4]), 4),
    "samples": records[:12],
}
out_dir = PROJECT / "experiments" / "confidence"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "semantic_calibration.json").write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
