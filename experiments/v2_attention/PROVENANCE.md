# PROVENANCE NOTE (Aug 2026 audit)

This directory NO LONGER contains the Gen-2 CE eval_results.json.
That file was silently overwritten by the RL evaluation (an output-dir
bug since fixed in eval_v2.py / eval_v2_clip.py — results now always save
next to the checkpoint that produced them).

CE numbers live in: priority0_report.json (metrics + CHAIR + calibration)
RL numbers live in: experiments/v2_attention_rl/eval_results.json
