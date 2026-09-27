"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Network,
  Layers,
  Type,
  Database,
  Hash,
  GitBranch,
  Activity,
  AlertCircle,
  CheckCircle2,
  Cpu,
  Box,
  Zap,
  TrendingUp,
  Wrench,
  Brain,
  Eye,
  Target,
  Gauge,
  ChevronDown,
  XCircle,
  Plug,
  ShieldCheck,
  Route,
  FlaskConical,
  ThumbsDown,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { apiClient, ApiError } from "@/lib/api";
import type { ModelInfoResponse } from "@/lib/types";

/* ================================================================== */
/* WINNING GENERATIONS                                                  */
/* ================================================================== */

const generations = [
  {
    tag: "Gen 1",
    name: "DenseNet201 + LSTM (TensorFlow)",
    status: "Baseline",
    tone: "slate" as const,
    icon: Network,
    bullets: [
      "Frozen DenseNet201 (ImageNet) → global-average-pooled 1920-d vector",
      "Embedding(256) + LSTM(256) + Dense softmax decoder (Keras)",
      "Word-level Keras tokenizer, greedy decoding",
      "Val BLEU-1 0.5334 · BLEU-4 0.1218 · ROUGE-L 0.2216 (beam-10)",
    ],
  },
  {
    tag: "Gen 2",
    name: "Attention v2 — spatial DenseNet + Bahdanau LSTM (PyTorch)",
    status: "Superseded",
    tone: "indigo" as const,
    icon: Layers,
    bullets: [
      "Kept DenseNet201 frozen but kept all 49 spatial positions (7×7×1920) instead of pooling to one vector",
      "Bahdanau additive attention lets the decoder look at the right region while emitting each word",
      "BPE-6k subword tokenizer, AdamW + label smoothing + scheduled sampling",
      "Beam search with GNMT length normalisation + repeat penalty",
      "Val BLEU-1 0.5649 · BLEU-4 0.1663 (+37% over Gen 1) · CIDEr-D 0.5397",
    ],
  },
  {
    tag: "Gen 2-RL",
    name: "Attention v2 + GRPO reinforcement learning",
    status: "Superseded",
    tone: "indigo" as const,
    icon: Target,
    bullets: [
      "GRPO: sample G=5 captions per image, reward = CIDEr-D, group-relative advantage",
      "EMA(0.999) weights served; fast fixed-DF CIDEr scorer validated vs pycocoevalcap",
      "Subset CIDEr-D 0.4165 → 0.4591 (+10.2%); full-split BLEU-1 0.5971",
      "Side-effect discovered: captions shorten (8.2 → 7.4 words) — the CIDEr precision bias",
    ],
  },
  {
    tag: "Gen 3",
    name: "CLIP ViT-B/16 patches + Bahdanau decoder",
    status: "Superseded",
    tone: "emerald" as const,
    icon: Eye,
    bullets: [
      "Feature swap ONLY: 196 patch tokens × 768 dims from CLIP (400M image-text pairs)",
      "Decoder byte-identical to Gen 2 except enc_dim 1920→768 (8.42M params)",
      "BLEU-1 0.6092 · BLEU-4 0.1755 · ROUGE-L 0.2747 — beat every prior model with no decoder change",
      "Proved: language-aligned pretraining > decoder architecture at 8k-image scale",
    ],
  },
  {
    tag: "Gen 3-RL",
    name: "CLIP + GRPO — the production champion",
    status: "Production",
    tone: "emerald" as const,
    icon: CheckCircle2,
    bullets: [
      "GRPO fine-tune on CIDEr-D, epoch 4 EMA weights served",
      "Full-split: BLEU-1 0.6559 · BLEU-4 0.1727 · ROUGE-L 0.2782 · CIDEr-D 0.5243",
      "Word precision 70.7% (vs 63.9% for CE) — every emitted word more likely correct",
      "Served with min-length guard (8 tokens), trailing-word trim, calibrated confidence (T=1.3)",
    ],
  },
];

/* ================================================================== */
/* TECHNIQUES — every win & failure, click to expand                    */
/* ================================================================== */

type Verdict = "win" | "fail" | "partial";

const techniques: {
  name: string;
  verdict: Verdict;
  category: string;
  what: string;
  why: string;
  metrics: [string, string][];
  verdictText: string;
}[] = [
  {
    name: "Spatial Attention v2 + BPE Tokenizer",
    verdict: "win",
    category: "Architecture",
    what: "Kept all 49 spatial feature positions (7×7×1920) instead of pooling to one vector. Bahdanau additive attention scores each position against the decoder state per word. BPE-6k subword tokenizer eliminates UNK tokens.",
    why: "Gen-1's global average pooling erased WHERE objects are; word-level tokenizer couldn't handle rare words.",
    metrics: [
      ["BLEU-1", "0.5334 → 0.5649"],
      ["BLEU-4", "0.1218 → 0.1663 (+37%)"],
      ["ROUGE-L", "0.2216 → 0.2548"],
      ["CIDEr-D", "— → 0.5397"],
    ],
    verdictText:
      "WIN. The decoder was the first bottleneck, not the features. Same frozen encoder, +37% BLEU-4.",
  },
  {
    name: "GRPO Reinforcement Learning (CIDEr-D)",
    verdict: "win",
    category: "Training",
    what: "Group Relative Policy Optimization: sample G=5 captions per image, reward each by CIDEr-D, reinforce above-group-mean samples (advantage = (r−mean)/std). EMA(0.999) weights served. Fast fixed-DF CIDEr scorer validated vs pycocoevalcap.",
    why: "Cross-entropy has exposure bias — training conditions on ground truth, inference on the model's own outputs.",
    metrics: [
      ["DenseNet subset CIDEr", "0.4165 → 0.4591 (+10.2%)"],
      ["CLIP subset CIDEr", "0.4859 → 0.5211 (+7.2%)"],
      ["Full-split BLEU-1", "0.6092 → 0.6559 (+4.7 pts)"],
      ["Word precision", "63.9% → 70.7%"],
    ],
    verdictText:
      "WIN — the biggest single-generation quality jump. Side-effect: captions shorten (CIDEr precision bias) and calibration degrades (ECE 0.012 → 0.254), both later addressed.",
  },
  {
    name: "CLIP ViT-B/16 Feature Transfer",
    verdict: "win",
    category: "Features",
    what: "Swap the encoder's output features only: OpenAI CLIP ViT-B/16 (trained on 400M image-text pairs) provides 196 patch tokens × 768 dims. Decoder unchanged except input dimension. Features precomputed as fp16 memmap (2.4 GB).",
    why: "After decoder, tokenizer and RL were all optimized, the frozen ImageNet encoder was the remaining bottleneck.",
    metrics: [
      ["BLEU-1", "0.5971 → 0.6092 (CE)"],
      ["BLEU-4", "0.1612 → 0.1755"],
      ["ROUGE-L", "0.2566 → 0.2747"],
      ["Greedy BLEU-1 alone", "0.6520 — beat ALL prior beam results"],
    ],
    verdictText:
      "WIN — the paper's central result. A pure feature swap beat every previous model on every metric. Language-aligned pretraining > architecture at 8k-image scale.",
  },
  {
    name: "Min-Length Guard + Word Trimming",
    verdict: "win",
    category: "Decoding",
    what: "Beam search suppresses the END token until 8 BPE tokens are generated; after decoding, trailing function words (the/of/on…) are trimmed. Both display-side only.",
    why: "GRPO's CIDEr reward makes captions stop early ('boy is jumping' vs 'boy is jumping into the water').",
    metrics: [
      ["Audit BLEU-1", "0.477 → 0.533"],
      ["Word precision", "70.7% (held)"],
      ["Avg length", "6.0 → 6.6 words"],
    ],
    verdictText:
      "WIN (deployed). Vindicated over two reward-side alternatives that both failed (see rl2/rl3 below).",
  },
  {
    name: "Incremental State-Carrying Beam Search",
    verdict: "win",
    category: "Serving",
    what: "Replaced the O(T²) prefix-recomputing beam search with one that carries each beam's LSTM hidden state between steps and batches all beams into a single forward per step — O(T). Bit-exact identical output.",
    why: "Original serving took 31.6 seconds per caption — unusable.",
    metrics: [
      ["Latency", "31,600 ms → ~420 ms (75×)"],
      ["With tempered confidence", "~590 ms (53×)"],
      ["Feature cache hits", "~135 ms"],
    ],
    verdictText:
      "WIN — core serving win. Same captions, 75× faster.",
  },
  {
    name: "fp16 CUDA-Only Encoder Precision",
    verdict: "win",
    category: "Serving",
    what: "CLIP visual trunk runs in half precision on GPU. On CPU, fp16 stays OFF because Windows emulates it in software.",
    why: "Halve the encoder's memory footprint on GPU.",
    metrics: [
      ["CPU fp16 encode", "23,000 ms (emulated!)"],
      ["CPU fp32 encode", "500 ms"],
      ["GPU fp16 encode", "~15 ms"],
    ],
    verdictText:
      "WIN — subtle Windows lesson: fp16 on CPU is emulated and catastrophic. fp32-on-CPU beats fp16-on-CPU by 46×.",
  },
  {
    name: "Turbopack Memory Eviction + OneDrive Fix",
    verdict: "win",
    category: "Infrastructure",
    what: "turbopackMemoryEviction:'full' + explicit turbopack.root + .next build dir junctioned off OneDrive to local disk + standalone output restricted to production + tee-pipe removed from dev script. ~11 GB of dead experiment artifacts purged.",
    why: "Next.js Turbopack compiling inside OneDrive grew to 17.4 GB committed memory, hit 15.91/15.91 GB RAM, NVMe queue depth 122, and OOM-crashed the machine repeatedly. NODE_OPTIONS couldn't cap it — Turbopack memory is Rust-side.",
    metrics: [
      ["Node footprint", "7,719 MB → 391–540 MB"],
      ["Frontend page load", "60 s+ → 0.1 s"],
      ["System RAM during builds", "15.9/15.9 GB → ~6 GB"],
    ],
    verdictText:
      "WIN — machine-killing crash resolved. Diagnosed with 2-second telemetry sampling (saved to CSV), not guesswork.",
  },
  {
    name: "Hybrid OOD Router + Margin Rule",
    verdict: "win",
    category: "Serving",
    what: "The already-loaded CLIP visual trunk zero-shot classifies each image against 6 text prompts (photo / screenshot / food / cartoon / painting / document). In-domain → trained model; else → local BLIP. Margin rule: BLIP only if prompt[0] loses by ≥0.02 cosine. Routed responses badged 'attention+blip'.",
    why: "The specialist hallucinates on non-Flickr images (cartoon → 'two people standing in the of water'); BLIP handles the open world. Routing reuses the encoder forward pass — near-zero cost.",
    metrics: [
      ["In-domain BLEU-1 (60-img audit)", "0.5739 vs BLIP 0.5033"],
      ["OOD routing recall", "100% (10/10 synthetic)"],
      ["False-OOD rate", "40.2% → 9.97% (margin fix, all 1,214 live)"],
      ["Latency penalty", "≈0 (reuses encoder pass)"],
    ],
    verdictText:
      "WIN (live-verified). Bare-argmax routing stole 40% of real photos for BLIP; the 0.02-cosine margin rule cut that to 10% while keeping perfect OOD recall. Found by auditing all 1,214 images, not a spot check.",
  },
  {
    name: "Tempered Confidence (T=1.3, display-only)",
    verdict: "win",
    category: "Serving",
    what: "After decoding finishes, a separate teacher-forced pass computes softmax(logits/1.3) for the confidence number shown in the UI. The decode path itself stays at T=1 — temperature never touches token selection, verified by bit-exact metric reproduction.",
    why: "GRPO sharpened the softmax: mean max-prob 0.571 vs top-1 accuracy 0.316. The UI was displaying 57% confidence on 32%-accurate output.",
    metrics: [
      ["ECE (raw, T=1)", "0.2542"],
      ["ECE (served, T=1.3)", "0.0862"],
      ["Example confidence", "0.478 → 0.322 (same caption)"],
      ["Metrics impact", "NONE — bit-exact reproduction"],
    ],
    verdictText:
      "WIN — the confidence number now means what it says. T*=1.3 was measured by teacher-forced NLL grid, not guessed.",
  },
  {
    name: "RL Calibration Regularizer (γ on mean max-prob)",
    verdict: "win",
    category: "Training",
    what: "Loss += 0.02 × mean(max_prob over rollout tokens), added inside GRPO. Penalizes over-peaked policies without touching token selection.",
    why: "First GRPO pass collapsed calibration (ECE 0.012 → 0.254). Needed a fix baked into the RL objective itself.",
    metrics: [
      ["Raw ECE (rl4 vs rl)", "0.2542 → 0.1669 (−34%)"],
      ["Top-1 token accuracy", "0.316 → 0.329 (+1.3 pts)"],
      ["Combined with T=1.3", "ECE 0.0306 (rl4) vs 0.0862 (champion)"],
    ],
    verdictText:
      "WIN — proven mechanism, reusable in every future RL run. (In rl4 it came bundled with a CHAIR reward that failed; the regularizer itself was not the problem.)",
  },
  {
    name: "DenseNet Partial Fine-Tune (P2)",
    verdict: "fail",
    category: "Training",
    what: "Unfroze denseblock4 of the encoder, differential learning rates (1e-5 encoder / 1e-4 decoder), val-CIDEr early stopping.",
    why: "Test whether adapting ImageNet weights to Flickr8K beats keeping them frozen.",
    metrics: [
      ["Val CIDEr-D", "0.4180 → 0.3989 (regressed)"],
      ["Early stop", "epoch 2 of 6"],
    ],
    verdictText:
      "FAIL. 8k images too few to adapt ImageNet features without forgetting; BatchNorm statistics destabilize. Frozen transfer won.",
  },
  {
    name: "Hard No-Repeat Trigram (P3a)",
    verdict: "fail",
    category: "Decoding",
    what: "Hard-block any trigram that already appears in the generated sequence (vs the existing soft penalty).",
    why: "Test whether repetition needed harder suppression.",
    metrics: [
      ["BLEU-1/2/3/4, ROUGE-L", "zero delta on all"],
    ],
    verdictText:
      "FAIL (no effect). The soft 2-gram penalty (log 0.4) already handles repetition at this data scale.",
  },
  {
    name: "Focal Loss Retrain (P3b)",
    verdict: "fail",
    category: "Training",
    what: "Focal loss (γ=2.0, α=0.25) from scratch on full data; three controlled attempts including an architecture-mismatch fix.",
    why: "Test whether hard-example weighting improves caption learning.",
    metrics: [
      ["Training loss", "decreased normally"],
      ["Greedy BLEU-1", "0.0038 — gibberish"],
      ["Attempts", "3 (val bug, arch fix, full data)"],
    ],
    verdictText:
      "FAIL catastrophically. Focal loss down-weights easy tokens — but at 8k images the easy tokens carry caption structure. Loss went down while output became noise.",
  },
  {
    name: "Mixed Reward R = CIDEr + 0.5·ROUGE-L (rl2)",
    verdict: "fail",
    category: "Training",
    what: "Added recall-friendly ROUGE-L to the GRPO reward, hoping to fix the length bias at the source.",
    why: "ROUGE-L rewards longer matching sequences — theoretically counters CIDEr's shortness bias.",
    metrics: [
      ["Best subset CIDEr", "0.5211 → 0.5159"],
      ["Full-split BLEU-1", "0.6559 → 0.6514"],
      ["Word precision", "70.7% → 68.0%"],
      ["Caption length", "6.0 → 6.0 (unchanged!)"],
    ],
    verdictText:
      "FAIL. ROUGE-L's LCS F-measure is itself precision-dominated on short captions — it added reward noise without curing the bias. Pure CIDEr stayed champion.",
  },
  {
    name: "Length-Scaled Reward R = CIDEr·√(len/ref_mean) (rl3)",
    verdict: "fail",
    category: "Training",
    what: "Multiplicatively scale the CIDEr reward by the square root of the length ratio (candidate vs reference mean), capping the ratio at 2.",
    why: "Harder version of the length fix: explicitly penalize dropping below reference length.",
    metrics: [
      ["Caption length", "6.0 → 8.1–8.8 (worked!)"],
      ["BLEU-4", "0.1727 → 0.1601 (−1.3 pts)"],
      ["BLEU-1", "0.6559 → 0.6538"],
    ],
    verdictText:
      "FAIL. Got the length back but paid precision for it — the tradeoff MOVED but didn't resolve. The decode-time min-len guard remained the better fix.",
  },
  {
    name: "Transformer Decoder (Gen 4)",
    verdict: "fail",
    category: "Architecture",
    what: "4-layer pre-norm Transformer (8 heads, d=512, FFN 2048, dropout 0.3) with causal self-attention + full cross-attention to all 196 CLIP patches. 23.4M params (+178% vs LSTM).",
    why: "Literature suggests transformers handle long spatial sequences better than LSTMs.",
    metrics: [
      ["Val loss", "3.700 → 3.794 (worse)"],
      ["Full-split BLEU-1", "0.6559 → 0.5546 (−10.1 pts)"],
      ["BLEU-4", "0.1727 → 0.1348"],
      ["Train loss", "fell to 2.19 while val stalled @ep12"],
    ],
    verdictText:
      "FAIL — textbook overfitting: 23M params memorize 8k images. The LSTM's recurrence is itself a regularizer. Transformers win at scale, not at 8k.",
  },
  {
    name: "Q-Former Learnable Queries",
    verdict: "fail",
    category: "Architecture",
    what: "K learnable query vectors (BLIP-2 style) would compress the 196 patch tokens into K≈32 'visual concepts' before the decoder attends.",
    why: "Compress context and reduce decoder attention load.",
    metrics: [["Status", "deprioritized after analysis — not run"]],
    verdictText:
      "REJECTED on analysis. BLIP-2's Q-Former was trained on 129M images; a from-scratch Q-Former on 8k adds random-init parameters exactly where data is scarcest. With only 196 tokens, attention isn't the bottleneck.",
  },
  {
    name: "Offline BLIP Knowledge Distillation",
    verdict: "fail",
    category: "Training",
    what: "BLIP generated 34,022 pseudo-captions (5 per training image, beam + sampling); the Gen-3 decoder retrained on original + synthetic captions (68,405 sequences). BLIP used offline only — discarded at inference.",
    why: "Distill the generalist's world knowledge into the specialist without serving BLIP.",
    metrics: [
      ["Val loss", "3.700 → 3.664 (BETTER)"],
      ["Full-split BLEU-1", "0.6092 → 0.4889 (−12 pts!!)"],
      ["BLEU-4", "0.1755 → 0.1153"],
    ],
    verdictText:
      "FAIL — the most instructive negative. Val loss improved while generation collapsed: BLIP's caption STYLE differs from Flickr8K references ('a door in the photo' vs 'door in a brick wall'). Better likelihood over the wrong distribution. Val loss ≠ quality.",
  },
  {
    name: "DistilBERT Beam Reranking",
    verdict: "fail",
    category: "Decoding",
    what: "Generate 5 diverse beam hypotheses, score each under distilbert-base-uncased pseudo-perplexity, pick the most fluent.",
    why: "Free inference-time quality gain from a general language prior.",
    metrics: [
      ["BLEU-1", "0.5351 → 0.5476 (+1.2)"],
      ["BLEU-4", "0.1348 → 0.1292 (−0.6)"],
      ["ROUGE-L", "0.2285 → 0.2272 (−0.1)"],
      ["Hypotheses changed", "812/1214 (67%)"],
    ],
    verdictText:
      "FAIL (net negative). Text-only fluency pulls AWAY from Flickr8K reference style — same failure family as distillation. External language knowledge ≠ this domain's style.",
  },
  {
    name: "Diverse Beam Search (λ=0.7/1.5)",
    verdict: "fail",
    category: "Decoding",
    what: "Penalize candidate beams sharing n-grams with already-selected beams (Vijayakumar et al. 2016), forcing diversity.",
    why: "5 similar beams waste capacity; diversity should cover more of the 5 references.",
    metrics: [
      ["λ=0: BLEU-1 0.5702", "BLEU-4 0.1510"],
      ["λ=0.7: BLEU-1 0.5559", "BLEU-4 0.1450"],
      ["λ=1.5: BLEU-1 0.5306", "BLEU-4 0.1432"],
    ],
    verdictText:
      "FAIL — monotonic harm. Flickr8K's 5 references are themselves similar, so diversity pushes beams toward low-probability phrasings that match nothing. Code kept (ATT_DBS_LAMBDA=0), default off.",
  },
  {
    name: "CHAIR-Aware GRPO Reward (rl4 + rl5)",
    verdict: "fail",
    category: "Training",
    what: "R = CIDEr-D − 0.05 × hallucinated_noun_count, with NLTK POS-tagging against precomputed reference noun sets. rl4 scored with greedy/300-img proxy; rl5 re-ran with REAL beam-5 full-split evaluation every epoch. rl4 also carried the calibration regularizer (γ=0.02).",
    why: "47.3% of champion captions contain ≥1 unsupported noun (CHAIR-lite audit) — hallucination was the biggest remaining quality defect.",
    metrics: [
      ["rl4 full-split: CHAIR-img", "47.3% → 49.5% (WORSE)"],
      ["rl5 full-split: CHAIR-img", "61.9%(CE) → 24.0% (via degeneration)"],
      ["rl5 CIDEr-D", "0.6207 → 0.3809 (crashed)"],
      ["rl5 length", "7.6 → 5.0 words (noun avoidance)"],
      ["rl5 ECE @T1.3", "0.3215 (calib reg overwhelmed)"],
    ],
    verdictText:
      "FAIL — and the failure mode is the finding: the model games the penalty by avoiding nouns and shortening (beam-5 then finds fluent captions inside the degenerate distribution). rl4's proxy eval also exposed the greedy/300-img vs beam-5/full-split measurement gap that made it LOOK like a fix during training. Positively: rl4 proved the calibration regularizer works, and rl5 fixed the eval gap permanently.",
  },
];

/* ================================================================== */
/* CORRECTED deployment numbers                                         */
/* ================================================================== */

const optimizations = [
  {
    icon: Gauge,
    stat: "53×",
    label: "faster inference",
    body: "31,600 ms → ~590 ms per caption. Incremental state-carrying beam search + tempered-confidence pass (75× before tempering was added; the honest current figure is 53×).",
  },
  {
    icon: Eye,
    stat: "~70%",
    label: "word precision",
    body: "Share of predicted words found in the human references — up from 64% pre-RL. GRPO made every word count.",
  },
  {
    icon: Zap,
    stat: "989 MB",
    label: "backend footprint",
    body: "fp16 encoder on CUDA only, capped torch threads, optional scorer off. Safe beside the dev server.",
  },
  {
    icon: Wrench,
    stat: "391 MB",
    label: "frontend compiler",
    body: "Turbopack eviction + build caches off OneDrive. Was 7.7 GB with machine-killing OOM crashes.",
  },
];

/* ================================================================== */
/* Backend / API documentation                                          */
/* ================================================================== */

const apiEndpoints = [
  {
    method: "GET",
    path: "/health",
    desc: "Liveness + which provider is active + weights loaded. Frontend polls this.",
    returns: '{status:"ok", model_loaded, provider:"attention", weights_loaded, backend}',
  },
  {
    method: "GET",
    path: "/model-info",
    desc: "Full architecture metadata of the active provider: encoder, dims, checkpoint epoch, val metrics, calibration note.",
    returns: "{ready, active_provider, encoder, encoder_feature_dim, decoder, vocab_size, max_length, training_metadata{note, val_metrics, checkpoint_epoch}}",
  },
  {
    method: "GET",
    path: "/providers",
    desc: "Lists available providers and the currently selected one.",
    returns: '{available:["auto","notebook-tensorflow","huggingface-blip","attention"], current, weights_loaded}',
  },
  {
    method: "POST",
    path: "/providers/{name}",
    desc: "Switch the active provider at runtime. 'attention' = the trained CLIP+GRPO model.",
    returns: '{status:"ok", provider, weights_loaded, message}',
  },
  {
    method: "POST",
    path: "/predict",
    desc: "Single-image captioning. Multipart file upload; optional ?provider= query to override for one request. Runs the FULL pipeline: CLIP encode → OOD router → (specialist beam-5 | BLIP) → tempered confidence.",
    returns: '{caption, inference_time:"416.9ms", confidence, semantic_score, semantic_pmi, provider:"attention"|"attention+blip", success, error}',
  },
  {
    method: "POST",
    path: "/predict-batch",
    desc: "Multi-image batch captioning, same pipeline per file.",
    returns: "{results:[{filename, caption, inference_time, confidence, success}], total, success_count}",
  },
];

const dataPath = [
  {
    n: 1,
    title: "Browser → Next.js (:3000)",
    sub: "User drops an image; dashboard calls fetch('/predict', FormData) — same origin, no CORS needed.",
    color: "from-sky-400 to-blue-500",
  },
  {
    n: 2,
    title: "Next.js rewrite → FastAPI (:8010)",
    sub: "next.config.ts rewrites /predict, /health, /model-info, /providers/* to the backend. (Port moved 8000 → 8010 after an unrelated service claimed 8000.)",
    color: "from-blue-400 to-indigo-500",
  },
  {
    n: 3,
    title: "ModelManager → AttentionProvider",
    sub: "Startup picks the best checkpoint (CLIP-RL > CLIP-CE > DenseNet-RL > DenseNet-CE); lazy-loads CLIP trunk + GRPO decoder + BPE tokenizer. /providers/{name} can hot-swap at runtime.",
    color: "from-indigo-400 to-purple-500",
  },
  {
    n: 4,
    title: "Hybrid OOD Router",
    sub: "Same CLIP trunk zero-shot scores the image against 6 domain prompts. Real photo (or marginal win <0.02 cosine) → specialist. Clearly non-photo (screenshot/food/cartoon, ≥0.02 margin) → BLIP fallback, response badged 'attention+blip'.",
    color: "from-purple-400 to-fuchsia-500",
  },
  {
    n: 5,
    title: "Specialist: incremental beam-5 decode",
    sub: "Bahdanau attention over 196 CLIP patches → LSTM(512) GRPO-tuned weights → min-len 8 guard → GNMT α=1.2 → trailing-word trim → tempered confidence (T=1.3).",
    color: "from-fuchsia-400 to-pink-500",
  },
  {
    n: 6,
    title: "Fallback: local BLIP",
    sub: "weights/blip snapshot (~1 GB, lazy-loaded on first OOD hit). Handles the open world the specialist never saw. ~2.5 s per caption.",
    color: "from-amber-400 to-orange-500",
  },
  {
    n: 7,
    title: "Response → UI badge",
    sub: "CaptionResult carries provider + confidence. The card shows 'CLIP + GRPO' or 'CLIP+GRPO → BLIP (auto-routed)' so you always know which model answered.",
    color: "from-emerald-400 to-teal-500",
  },
];

const healthChecks = [
  {
    icon: ShieldCheck,
    title: "Routing health",
    body: "In-domain stays with the specialist (90.0% of 1,214 val images after the margin fix); OOD routes to BLIP with 100% recall on the synthetic battery. Watch the /predict response's provider field: 'attention+blip' means auto-routing fired.",
  },
  {
    icon: Gauge,
    title: "Latency health",
    body: "Warm specialist: ~450–620 ms. BLIP fallback: ~2.5 s (lazy-loads on first OOD hit, ~1.1 GB). Cold start after restart: first call ~20 s (model load + OneDrive hydration).",
  },
  {
    icon: ShieldCheck,
    title: "Confidence meaning",
    body: "Since T=1.3 tempering, the confidence number is calibrated: it reflects true per-token accuracy instead of GRPO's over-peaked softmax. ~0.3–0.5 is a normal good caption; the old 0.5+ readings were inflated.",
  },
  {
    icon: Activity,
    title: "Known operational quirks",
    body: "Port 8000 is claimed by an unrelated docuflow service — backend lives on 8010. A 'notebook-tensorflow' provider can be selected accidentally from Settings and will silently serve the weak Gen-1 model; /health always tells the truth. BLIP scorer stays off (ENABLE_BLIP_SCORE=0) unless semantic scores are explicitly needed.",
  },
];

function toneClasses(tone: "slate" | "indigo" | "emerald") {
  switch (tone) {
    case "emerald":
      return {
        chip: "text-emerald-300 bg-emerald-500/15 ring-emerald-500/40",
        border: "border-emerald-500/50 bg-emerald-500/[0.04]",
        dot: "bg-emerald-400",
      };
    case "indigo":
      return {
        chip: "text-indigo-300 bg-indigo-500/15 ring-indigo-500/40",
        border: "border-indigo-500/30",
        dot: "bg-indigo-400",
      };
    default:
      return {
        chip: "text-slate-300 bg-slate-500/15 ring-slate-500/40",
        border: "border-border/40",
        dot: "bg-slate-400",
      };
  }
}

export function ModelInfoView() {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [openTech, setOpenTech] = useState<string | null>(null);

  useEffect(() => {
    apiClient
      .getModelInfo()
      .then(setInfo)
      .catch((e) => {
        if (e instanceof ApiError) setError(e.detail);
        else setError(String(e));
      });
  }, []);

  const isClip = (info?.encoder || "").toLowerCase().includes("clip");
  const wins = techniques.filter((t) => t.verdict === "win");
  const fails = techniques.filter((t) => t.verdict === "fail");

  return (
    <div className="space-y-10">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="font-display text-3xl font-bold tracking-tight">
          Model Info
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Production champion (CLIP+GRPO), the complete technique record —
          every win and failure with data — and the full backend/API pathway.
        </p>
      </motion.div>

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Couldn’t load model info</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Status cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatusCard
          icon={info?.ready ? CheckCircle2 : AlertCircle}
          label="Status"
          value={info?.ready ? "Ready" : "Loading…"}
          tone={info?.ready ? "emerald" : "amber"}
        />
        <StatusCard
          icon={Brain}
          label="Provider"
          value={isClip ? "CLIP + GRPO (champion)" : info?.provider || "—"}
          tone={isClip ? "emerald" : "indigo"}
        />
        <StatusCard
          icon={Box}
          label="Weights"
          value={
            info?.weights_loaded
              ? `Epoch ${epochOf(info)} loaded`
              : "Fallback"
          }
          tone={info?.weights_loaded ? "emerald" : "amber"}
        />
        <StatusCard
          icon={Activity}
          label="Runtime"
          value="PyTorch + open_clip (no TF on hot path)"
          tone="purple"
        />
      </div>

      {/* ---------------- Champion detail ---------------- */}
      <section>
        <SectionHeading
          icon={CheckCircle2}
          title="The Production Champion — CLIP ViT-B/16 + GRPO"
          sub="Everything about the model currently serving, with its verified numbers."
        />
        <Card className="mt-5 border-emerald-500/40 bg-emerald-500/[0.03] p-6 backdrop-blur-sm ring-1 ring-emerald-500/20">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
            <Stat label="BLEU-1" value="0.6559" />
            <Stat label="BLEU-4" value="0.1727" />
            <Stat label="ROUGE-L" value="0.2782" />
            <Stat label="CIDEr-D" value="0.5243" />
            <Stat label="Word precision" value="70.7%" />
            <Stat label="CHAIR-img rate" value="47.3%" />
            <Stat label="Confidence ECE" value="0.0862 (T=1.3)" />
            <Stat label="Median latency" value="~590 ms" />
            <Stat
              label="Checkpoint"
              value={info ? `epoch ${epochOf(info)} (EMA)` : "—"}
            />
            <Stat label="Params" value="8.42M" />
          </div>
          <div className="mt-5 space-y-2 text-sm leading-relaxed text-muted-foreground">
            <p>
              <b className="text-foreground">Architecture:</b> frozen CLIP
              ViT-B/16 vision trunk (196 patches × 768 dims, fp16 on CUDA) →
              Bahdanau additive attention → LSTM(512) decoder with GRPO-tuned
              EMA weights → BPE-6k output.
            </p>
            <p>
              <b className="text-foreground">Decoding:</b> incremental
              state-carrying beam search (beam 5, GNMT α=1.2, soft 2-gram
              penalty), min-length guard (8 tokens), trailing-function-word
              trim. Confidence computed separately at T=1.3 — decode path
              untouched.
            </p>
            <p>
              <b className="text-foreground">Journey to here:</b> Gen 1
              (DenseNet+LSTM, BLEU-1 0.5334) → Gen 2 (spatial attention,
              0.5649) → Gen 2-RL (GRPO, 0.5971) → Gen 3 (CLIP features,
              0.6092) → <b className="text-emerald-300">Gen 3-RL (GRPO on
              CLIP, 0.6559)</b>. Total: +22.9% BLEU-1, +41.8% BLEU-4.
            </p>
          </div>
        </Card>
      </section>

      {/* ---------------- Journey ---------------- */}
      <section>
        <SectionHeading
          icon={GitBranch}
          title="Generation history"
          sub="Each generation kept what worked and replaced what bottlenecked."
        />
        <div className="mt-5 space-y-4">
          {generations.map((g, i) => {
            const t = toneClasses(g.tone);
            return (
              <motion.div
                key={g.tag}
                initial={{ opacity: 0, y: 14 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: i * 0.05 }}
              >
                <Card
                  className={`relative overflow-hidden p-5 backdrop-blur-sm ${t.border} ${
                    g.status === "Production" ? "ring-1 ring-emerald-500/30" : ""
                  }`}
                >
                  <div className="flex flex-wrap items-center gap-2">
                    <span
                      className={`rounded-md px-2 py-0.5 font-mono text-[11px] font-bold uppercase tracking-wider ring-1 ring-inset ${t.chip}`}
                    >
                      {g.tag}
                    </span>
                    <h3 className="font-display text-lg font-bold">
                      {g.name}
                    </h3>
                    {g.status === "Production" && (
                      <Badge className="ml-auto gap-1 bg-emerald-500/15 text-emerald-300 ring-1 ring-inset ring-emerald-500/40">
                        <span className={`h-1.5 w-1.5 rounded-full ${t.dot}`} />
                        Live now
                      </Badge>
                    )}
                  </div>
                  <ul className="mt-3 space-y-1.5">
                    {g.bullets.map((b) => (
                      <li
                        key={b.slice(0, 24)}
                        className="flex gap-2 text-sm leading-relaxed text-muted-foreground"
                      >
                        <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-muted-foreground/60" />
                        {b}
                      </li>
                    ))}
                  </ul>
                </Card>
              </motion.div>
            );
          })}
        </div>
      </section>

      {/* ---------------- Techniques (expandable) ---------------- */}
      <section>
        <SectionHeading
          icon={FlaskConical}
          title="Techniques — every win & failure, with data"
          sub="Click any technique to expand its full details, metrics and verdict. 10 wins · 10 documented failures."
        />
        <div className="mt-4 flex flex-wrap gap-2">
          <Badge className="gap-1 bg-emerald-500/15 text-emerald-300 ring-1 ring-inset ring-emerald-500/40">
            <CheckCircle2 className="h-3 w-3" /> {wins.length} wins
          </Badge>
          <Badge className="gap-1 bg-rose-500/15 text-rose-300 ring-1 ring-inset ring-rose-500/40">
            <XCircle className="h-3 w-3" /> {fails.length} failures (documented)
          </Badge>
        </div>

        <div className="mt-5 space-y-2.5">
          {techniques.map((t, i) => {
            const open = openTech === t.name;
            const isWin = t.verdict === "win";
            return (
              <motion.div
                key={t.name}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.3, delay: Math.min(i * 0.02, 0.2) }}
              >
                <Card
                  className={`overflow-hidden backdrop-blur-sm transition-colors ${
                    open
                      ? isWin
                        ? "border-emerald-500/40"
                        : "border-rose-500/40"
                      : "border-border/40 hover:border-border"
                  }`}
                >
                  <button
                    onClick={() => setOpenTech(open ? null : t.name)}
                    className="flex w-full items-center gap-3 p-4 text-left"
                  >
                    <div
                      className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ring-1 ring-inset ${
                        isWin
                          ? "bg-emerald-500/10 text-emerald-400 ring-emerald-500/30"
                          : "bg-rose-500/10 text-rose-400 ring-rose-500/30"
                      }`}
                    >
                      {isWin ? (
                        <CheckCircle2 className="h-5 w-5" />
                      ) : (
                        <XCircle className="h-5 w-5" />
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <h3 className="font-display font-semibold">
                          {t.name}
                        </h3>
                        <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] uppercase tracking-wider text-muted-foreground">
                          {t.category}
                        </span>
                      </div>
                      <p
                        className={`mt-0.5 text-xs font-medium ${
                          isWin ? "text-emerald-400" : "text-rose-400"
                        }`}
                      >
                        {isWin ? "WIN" : "FAIL"} —{" "}
                        {t.metrics[0]
                          ? `${t.metrics[0][0]}: ${t.metrics[0][1]}`
                          : t.verdictText.slice(0, 60)}
                      </p>
                    </div>
                    <ChevronDown
                      className={`h-5 w-5 shrink-0 text-muted-foreground transition-transform ${
                        open ? "rotate-180" : ""
                      }`}
                    />
                  </button>

                  <AnimatePresence initial={false}>
                    {open && (
                      <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.25 }}
                      >
                        <div className="border-t border-border/40 px-4 pb-4 pt-3">
                          <p className="text-sm leading-relaxed text-muted-foreground">
                            <b className="text-foreground">What: </b>
                            {t.what}
                          </p>
                          <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                            <b className="text-foreground">Why tried: </b>
                            {t.why}
                          </p>
                          <div className="mt-3 overflow-x-auto rounded-lg border border-border/40">
                            <table className="w-full text-left text-xs">
                              <thead>
                                <tr className="border-b border-border/60 bg-muted/40 text-[10px] uppercase tracking-wider text-muted-foreground">
                                  <th className="px-3 py-1.5 font-medium">
                                    Metric
                                  </th>
                                  <th className="px-3 py-1.5 font-medium">
                                    Value
                                  </th>
                                </tr>
                              </thead>
                              <tbody className="font-mono">
                                {t.metrics.map(([k, v]) => (
                                  <tr
                                    key={k}
                                    className="border-b border-border/30 last:border-0"
                                  >
                                    <td className="px-3 py-1.5">{k}</td>
                                    <td className="px-3 py-1.5">{v}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                          <p
                            className={`mt-3 rounded-lg border-l-4 bg-card/60 p-3 text-sm leading-relaxed ${
                              isWin
                                ? "border-emerald-500 bg-emerald-500/[0.06]"
                                : "border-rose-500 bg-rose-500/[0.06]"
                            }`}
                          >
                            <b
                              className={
                                isWin ? "text-emerald-300" : "text-rose-300"
                              }
                            >
                              Verdict:{" "}
                            </b>
                            {t.verdictText}
                          </p>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </Card>
              </motion.div>
            );
          })}
        </div>
      </section>

      {/* ---------------- Metrics comparison ---------------- */}
      <section>
        <SectionHeading
          icon={TrendingUp}
          title="Measured progress (full validation split)"
          sub="Same 1,214 images, same evaluation protocol across every generation."
        />
        <Card className="mt-5 overflow-x-auto border-border/40 bg-card/40 p-5 backdrop-blur-sm">
          <table className="w-full min-w-[640px] text-left text-sm">
            <thead>
              <tr className="border-b border-border/60 text-xs uppercase tracking-wider text-muted-foreground">
                <th className="pb-2 pr-4 font-medium">Generation</th>
                <th className="pb-2 pr-4 font-medium">BLEU-1</th>
                <th className="pb-2 pr-4 font-medium">BLEU-4</th>
                <th className="pb-2 pr-4 font-medium">ROUGE-L</th>
                <th className="pb-2 font-medium">Word precision*</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {(
                [
                  ["Gen 1 — DenseNet+LSTM", "0.5334", "0.1218", "0.2216", "—"],
                  ["Gen 2 — Attention v2 (CE)", "0.5649", "0.1663", "0.2548", "—"],
                  ["Gen 2 — Attention v2 (RL)", "0.5971", "0.1612", "0.2566", "~64%"],
                  ["Gen 3 — CLIP (CE)", "0.6092", "0.1755", "0.2747", "63.9%"],
                  ["Gen 3 — CLIP+GRPO ★", "0.6559", "0.1727", "0.2782", "70.7%"],
                ] as const
              ).map((row, i) => (
                <tr
                  key={row[0]}
                  className={`border-b border-border/30 last:border-0 ${
                    i === 4 ? "font-semibold text-emerald-300" : ""
                  }`}
                >
                  {row.map((c, j) => (
                    <td key={j} className="py-2 pr-4">
                      {c}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-xs text-muted-foreground">
            * share of predicted words appearing in any of the 5 references,
            measured on a random 30-image audit. ★ served in production.
          </p>
        </Card>
      </section>

      {/* ---------------- Current pipeline ---------------- */}
      <section>
        <SectionHeading
          icon={Network}
          title="Current Pipeline (live)"
          sub="End-to-end flow as served right now."
        />
        <Card className="mt-5 overflow-hidden border-border/40 bg-card/40 p-6 backdrop-blur-sm">
          <div className="space-y-3">
            <PipelineRow
              n={1}
              label="Input image"
              sub="224 × 224 RGB, CLIP normalisation"
              color="from-blue-400 to-cyan-500"
            />
            <PipelineArrow label="patchify 16×16 → 196 tokens" />
            <PipelineRow
              n={2}
              label="CLIP ViT-B/16 vision trunk (frozen, fp16 on CUDA)"
              sub="language-aligned patch embeddings"
              color="from-emerald-400 to-teal-500"
            />
            <PipelineArrow label="196 × 768 patch matrix" />
            <PipelineRow
              n={3}
              label="Bahdanau attention over 196 patches"
              sub="learns WHERE to look for every token"
              color="from-indigo-400 to-purple-500"
            />
            <PipelineArrow label="context vector ⊕ token embedding" />
            <PipelineRow
              n={4}
              label="LSTM(512) decoder — GRPO-tuned weights"
              sub="incremental state-carrying beam search (beam 5, GNMT α=1.2)"
              color="from-fuchsia-400 to-pink-500"
            />
            <PipelineArrow label="min-length ≥8 guard + repeat penalty + trim" />
            <PipelineRow
              n={5}
              label="Caption out"
              sub="BPE-decoded, complete sentence"
              color="from-amber-400 to-orange-500"
            />
          </div>
        </Card>
      </section>

      {/* ---------------- Backend / API ---------------- */}
      <section>
        <SectionHeading
          icon={Plug}
          title="Backend, API & data pathway"
          sub="Every endpoint, the full request journey, and how the frontend reaches the models."
        />
        <Card className="mt-5 overflow-hidden border-border/40 bg-card/40 p-6 backdrop-blur-sm">
          <div className="space-y-3">
            {dataPath.map((s, i) => (
              <div key={s.n}>
                <div className="flex items-start gap-3">
                  <div
                    className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br ${s.color} text-sm font-bold text-white shadow-md`}
                  >
                    {s.n}
                  </div>
                  <div className="flex-1 rounded-lg border border-border/40 bg-card/30 px-4 py-2.5">
                    <p className="text-sm font-semibold">{s.title}</p>
                    <p className="text-xs leading-relaxed text-muted-foreground">
                      {s.sub}
                    </p>
                  </div>
                </div>
                {i < dataPath.length - 1 && (
                  <div className="ml-[17px] flex h-5 items-center">
                    <div className="h-full w-px bg-border" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </Card>

        <Card className="mt-5 overflow-x-auto border-border/40 bg-card/40 p-5 backdrop-blur-sm">
          <h3 className="font-display text-lg font-bold">API endpoints</h3>
          <div className="mt-3 space-y-2.5">
            {apiEndpoints.map((e) => (
              <div
                key={e.path + e.method}
                className="rounded-lg border border-border/40 bg-card/30 p-3"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-bold ${
                      e.method === "GET"
                        ? "bg-sky-500/15 text-sky-300"
                        : "bg-emerald-500/15 text-emerald-300"
                    }`}
                  >
                    {e.method}
                  </span>
                  <code className="font-mono text-sm">{e.path}</code>
                </div>
                <p className="mt-1.5 text-sm text-muted-foreground">{e.desc}</p>
                <p className="mt-1 font-mono text-[10px] leading-relaxed text-muted-foreground/70">
                  → {e.returns}
                </p>
              </div>
            ))}
          </div>
        </Card>

        <div className="mt-5 grid grid-cols-1 gap-4 lg:grid-cols-2">
          {healthChecks.map((h) => (
            <Card
              key={h.title}
              className="border-border/40 bg-card/40 p-5 backdrop-blur-sm"
            >
              <div className="flex items-start gap-3">
                <div className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-500/20 to-teal-500/20 ring-1 ring-inset ring-emerald-500/30">
                  <h.icon className="h-5 w-5 text-emerald-400" />
                </div>
                <div>
                  <h3 className="font-display font-semibold">{h.title}</h3>
                  <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
                    {h.body}
                  </p>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </section>

      {/* ---------------- Optimizations ---------------- */}
      <section>
        <SectionHeading
          icon={Gauge}
          title="Optimizations live right now"
          sub="Speed and stability work applied to serving and dev infrastructure."
        />
        <div className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {optimizations.map((o, i) => (
            <motion.div
              key={o.label}
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.35, delay: i * 0.05 }}
            >
              <Card className="h-full border-border/40 bg-card/40 p-5 backdrop-blur-sm">
                <o.icon className="h-5 w-5 text-emerald-400" />
                <p className="mt-3 font-display text-2xl font-bold tracking-tight text-emerald-300">
                  {o.stat}
                </p>
                <p className="text-xs uppercase tracking-wider text-muted-foreground">
                  {o.label}
                </p>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  {o.body}
                </p>
              </Card>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ---------------- Live hyperparameters ---------------- */}
      {info && (
        <section>
          <SectionHeading
            icon={Cpu}
            title="Live hyperparameters (from backend)"
            sub="Read straight from the running server — reflects whatever provider is active."
          />
          <Card className="mt-5 border-border/40 bg-card/40 p-6 backdrop-blur-sm">
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
              <Stat
                label="Vocabulary (BPE)"
                value={info.vocab_size.toLocaleString()}
              />
              <Stat label="Max length" value={`${info.max_length} tokens`} />
              <Stat label="Min length guard" value="8 tokens" />
              <Stat label="Confidence temp" value="T = 1.3" />
              <Stat label="Embedding dim" value={`${info.embed_dim}`} />
              <Stat label="Decoder dim" value="512" />
              <Stat label="Encoder dim" value={`${info.encoder_feature_dim}`} />
              <Stat label="Image size" value={`${info.image_size}px`} />
              <Stat label="Decoding" value="beam 5 · α 1.2" />
              <Stat label="Router margin" value="0.02 cosine" />
            </div>
          </Card>
        </section>
      )}

      {/* Raw training metadata */}
      {info?.training_metadata &&
        Object.keys(info.training_metadata).length > 0 && (
          <Card className="border-border/40 bg-card/40 p-6 backdrop-blur-sm">
            <h2 className="font-display text-xl font-bold">
              Raw training metadata (from live checkpoint)
            </h2>
            <pre className="mt-4 overflow-x-auto rounded-lg border border-border/40 bg-muted/30 p-4 font-mono text-xs">
              {JSON.stringify(info.training_metadata, null, 2)}
            </pre>
          </Card>
        )}
    </div>
  );
}

function epochOf(info: ModelInfoResponse): string {
  const tm = info.training_metadata as Record<string, unknown>;
  const ep = tm?.checkpoint_epoch;
  return typeof ep === "number" ? String(ep) : "?";
}

function SectionHeading({
  icon: Icon,
  title,
  sub,
}: {
  icon: typeof GitBranch;
  title: string;
  sub: string;
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="mt-0.5 inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500/20 to-purple-500/20 ring-1 ring-inset ring-indigo-500/30">
        <Icon className="h-4.5 w-4.5 text-indigo-400" />
      </div>
      <div>
        <h2 className="font-display text-xl font-bold">{title}</h2>
        <p className="text-sm text-muted-foreground">{sub}</p>
      </div>
    </div>
  );
}

function StatusCard({
  icon: Icon,
  label,
  value,
  tone,
}: {
  icon: typeof CheckCircle2;
  label: string;
  value: string;
  tone: "emerald" | "indigo" | "purple" | "amber";
}) {
  const tones: Record<string, string> = {
    emerald: "text-emerald-400 bg-emerald-500/10 ring-emerald-500/30",
    indigo: "text-indigo-400 bg-indigo-500/10 ring-indigo-500/30",
    purple: "text-purple-400 bg-purple-500/10 ring-purple-500/30",
    amber: "text-amber-400 bg-amber-500/10 ring-amber-500/30",
  };
  return (
    <Card className="border-border/40 bg-card/40 p-4 backdrop-blur-sm">
      <div className="flex items-center gap-2">
        <div
          className={`flex h-8 w-8 items-center justify-center rounded-lg ring-1 ring-inset ${tones[tone]}`}
        >
          <Icon className="h-4 w-4" />
        </div>
        <span className="text-xs uppercase tracking-wider text-muted-foreground">
          {label}
        </span>
      </div>
      <p className="mt-3 truncate font-mono text-sm font-semibold">{value}</p>
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border/40 bg-card/30 p-3">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="mt-1 font-mono text-sm font-semibold">{value}</p>
    </div>
  );
}

function PipelineRow({
  n,
  label,
  sub,
  color,
}: {
  n: number;
  label: string;
  sub: string;
  color: string;
}) {
  return (
    <div className="flex items-center gap-3">
      <div
        className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br ${color} text-sm font-bold text-white shadow-md`}
      >
        {n}
      </div>
      <div className="flex-1 rounded-lg border border-border/40 bg-card/30 px-4 py-2.5">
        <p className="text-sm font-semibold">{label}</p>
        <p className="text-xs text-muted-foreground">{sub}</p>
      </div>
    </div>
  );
}

function PipelineArrow({ label }: { label: string }) {
  return (
    <div className="ml-5 flex items-center gap-2 pl-5">
      <div className="flex h-6 w-px flex-col justify-center bg-border">
        <motion.div
          animate={{ y: [0, 8, 0], opacity: [1, 0.3, 1] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
          className="h-2 w-px bg-indigo-400"
        />
      </div>
      <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
    </div>
  );
}
