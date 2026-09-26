"use client";

import { motion } from "framer-motion";
import { Layers, Hash, Database, GitBranch, Network, Type } from "lucide-react";
import { Card } from "@/components/ui/card";

const sections = [
  {
    icon: Network,
    title: "Encoder — CLIP ViT-B/16",
    body: "A frozen OpenAI CLIP ViT-B/16 vision trunk, pretrained on 400M image-text pairs, converts each 224×224 image into 196 patch tokens of 768 dims. Unlike global average pooling, every spatial position survives into the decoder. Features are precomputed once (fp16 memmap) and the trunk runs fp16 on CUDA / fp32 on CPU.",
    points: [
      "Backbone: CLIP ViT-B/16 (400M pairs)",
      "Output: 196 × 768-dim patch tokens",
      "Trainable: No (frozen; features cached)",
    ],
  },
  {
    icon: Layers,
    title: "Decoder — Attention LSTM + GRPO",
    body: "An 8.42M-parameter PyTorch decoder: Embedding(512) → Bahdanau additive attention over all 196 patches → LSTM(512) → softmax over the BPE vocabulary. Trained with cross-entropy, then fine-tuned by GRPO against a CIDEr-D reward (G=5 rollouts, group-relative advantage, EMA weights served).",
    points: [
      "Attention: Bahdanau additive",
      "LSTM units: 512 · Params: 8.42M",
      "RL: GRPO, CIDEr-D reward, EMA 0.999",
    ],
  },
  {
    icon: Type,
    title: "Tokenizer — BPE-6k",
    body: "A byte-pair encoding vocabulary of ~6,000 subword tokens replaces the old word-level tokenizer, so rare words compose from pieces instead of collapsing to UNK. The same tokenizer encodes references and decodes predictions.",
    points: [
      "Vocabulary: ~6,000 subwords",
      "No UNK tokens — rare words compose",
      "Shared train/inference encoding",
    ],
  },
  {
    icon: Database,
    title: "Training Dataset — Flickr8K",
    body: "Flickr8K: 8,091 photographs, each with 5 human captions (40,455 total), split 85/15 into 6,877 train / 1,214 validation. All model improvements were measured on the same fixed validation split with the same evaluation protocol.",
    points: [
      "Images: 8,091 · Captions: 40,455",
      "Train/val split: 85% / 15%",
      "Eval: beam-5, GNMT α=1.2, seed 42",
    ],
  },
  {
    icon: Hash,
    title: "Decoding — Beam-5 + Guards",
    body: "Incremental state-carrying beam search (beam 5) with GNMT length normalisation (α=1.2) and a soft 2-gram repeat penalty. A min-length guard suppresses END until 8 tokens — countering GRPO's shortness bias — and trailing function words are trimmed after decode.",
    points: [
      "Beam 5 · GNMT α=1.2 · repeat penalty",
      "Min-length guard: 8 BPE tokens",
      "O(T) state-carrying: ~590 ms warm",
    ],
  },
  {
    icon: GitBranch,
    title: "Serving — Router + Calibration",
    body: "At request time the CLIP trunk doubles as a zero-shot domain router: real photos stay with the specialist; cartoons/screenshots/food photos route to local BLIP (margin rule ≥0.02 cosine). Displayed confidence is tempered at T=1.3 in a separate pass — the decode path never sees it.",
    points: [
      "OOD router: 100% recall, 10% false-OOD",
      "Confidence: ECE 0.0862 at T=1.3",
      "Fallback: local BLIP (~2.5 s, lazy)",
    ],
  },
];

export function ModelSection() {
  return (
    <section className="relative py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-3xl text-center">
          <motion.span
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="inline-block rounded-full bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-400"
          >
            The Model
          </motion.span>
          <motion.h2
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mt-4 font-display text-4xl font-bold tracking-tight sm:text-5xl"
          >
            Three generations of research,
            <br />
            <span className="text-gradient">one production model</span>
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="mt-4 text-lg text-muted-foreground"
          >
            Started from the classic DenseNet+LSTM notebook recipe, then
            rebuilt: spatial attention, BPE, GRPO reinforcement learning and
            CLIP features — every step measured on the same validation split.
          </motion.p>
        </div>

        <div className="mt-16 grid grid-cols-1 gap-6 lg:grid-cols-2">
          {sections.map((s, i) => (
            <motion.div
              key={s.title}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.5, delay: (i % 2) * 0.1 }}
            >
              <Card className="h-full border-border/40 bg-card/40 p-6 backdrop-blur-sm">
                <div className="flex items-start gap-4">
                  <div className="inline-flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 ring-1 ring-inset ring-indigo-500/30">
                    <s.icon className="h-6 w-6 text-indigo-400" />
                  </div>
                  <div className="flex-1">
                    <h3 className="font-display text-lg font-semibold">
                      {s.title}
                    </h3>
                    <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                      {s.body}
                    </p>
                    <ul className="mt-4 space-y-1.5">
                      {s.points.map((p) => (
                        <li
                          key={p}
                          className="flex items-center gap-2 text-xs text-muted-foreground"
                        >
                          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                          <code className="font-mono text-foreground/80">{p}</code>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
