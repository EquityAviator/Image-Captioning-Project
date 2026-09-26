"""Validate PMI-based semantic scoring: true caption vs wrong caption on 6 val images."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
from PIL import Image

from inference import blip_scorer as bs

bs._load()
BACKEND = Path(__file__).resolve().parent


def cond(image, caption):
    inp = bs._processor(images=image, text=caption, return_tensors="pt",
                        padding=True).to(bs._device)
    with torch.no_grad():
        out = bs._model(**inp)
    lp = torch.log_softmax(out.logits[:, :-1, :].float(), -1)
    tok = lp.gather(-1, inp["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
    return float(tok.mean())


data = pd.read_csv(BACKEND.parent / "dataset" / "captions.txt")
gray = Image.new("RGB", (224, 224), (128, 128, 128))
groups = list(data.groupby("image"))[200:206]
wins = n = 0
raw_wins = 0
for img_name, grp in groups:
    img = Image.open(BACKEND.parent / "dataset" / "Images" / img_name).convert("RGB").resize((224, 224))
    true_cap = grp["caption"].iloc[0].lower()
    other_cap = data[data["image"] != img_name]["caption"].iloc[500].lower()
    pmi_true = cond(img, true_cap) - cond(gray, true_cap)
    pmi_wrong = cond(img, other_cap) - cond(gray, other_cap)
    raw_true = cond(img, true_cap)
    raw_wrong = cond(img, other_cap)
    n += 1
    wins += pmi_true > pmi_wrong
    raw_wins += raw_true > raw_wrong
    print(f"{img_name[:22]:22s} PMI true={pmi_true:+.3f} wrong={pmi_wrong:+.3f} "
          f"| RAW true={raw_true:+.3f} wrong={raw_wrong:+.3f}")
print(f"\nPMI  separated true/wrong on {wins}/{n}")
print(f"RAW  separated true/wrong on {raw_wins}/{n}")
