"use client";

import { motion } from "framer-motion";
import { Layers, Hash, Database, GitBranch, Network, Type } from "lucide-react";
import { Card } from "@/components/ui/card";

const sections = [
  {
    icon: Network,
    title: "Encoder — DenseNet201",
    body: "The encoder is a DenseNet201 convolutional network pretrained on ImageNet. We remove the final classification head (model.layers[-2]) and apply global average pooling, yielding a 1920-dimensional feature vector for every 224×224 input image. The encoder weights are frozen during decoder training — only the decoder learns.",
    points: [
      "Backbone: DenseNet201 (ImageNet weights)",
      "Output: 1920-dim feature vector",
      "Trainable: No (frozen at inference)",
    ],
  },
  {
    icon: Layers,
    title: "Decoder — LSTM",
    body: "The decoder is a custom Keras model that takes two inputs: the image feature vector and a partial caption sequence. The image features pass through a Dense(256, relu) layer and are reshaped to (1, 256), then concatenated with the embedded caption tokens. A 256-unit LSTM consumes the merged sequence, and a residual addition merges the LSTM output back with the image features.",
    points: [
      "Embedding dim: 256",
      "LSTM units: 256",
      "Dense layers: 256 → 128 → vocab_size",
      "Dropout: 0.5 (×2)",
    ],
  },
  {
    icon: Type,
    title: "Embedding Layer",
    body: "Each token in the partial caption is mapped to a 256-dim dense vector via a learned embedding. The embedding is trained jointly with the rest of the decoder, so semantically related words end up close in embedding space.",
    points: [
      "Vocabulary size: ~8,000 tokens",
      "Embedding dimension: 256",
      "mask_zero = False (manual padding)",
    ],
  },
  {
    icon: Database,
    title: "Training Dataset — Flickr8K",
    body: "Flickr8K contains 8,091 photographs crawled from Flickr, each annotated with 5 human-written captions (40,455 captions total). After lowercasing, removing non-alphabetic characters, and wrapping with startseq/endseq tokens, captions are tokenised with Keras’ Tokenizer.",
    points: [
      "Images: 8,091",
      "Captions: 40,455",
      "Train/val split: 85% / 15%",
      "Preprocessing: lowercase + startseq/endseq",
    ],
  },
  {
    icon: Hash,
    title: "Tokenizer & Vocabulary",
    body: "The Keras Tokenizer is fit on the training captions, producing a word→index mapping. Inference uses the same tokenizer to convert the running caption string into a padded integer sequence and to map predicted indices back to words via idx_to_word.",
    points: [
      "Tokeniser: keras.preprocessing.text.Tokenizer",
      "Special tokens: startseq / endseq",
      "Padding: post, to max_length",
    ],
  },
  {
    icon: GitBranch,
    title: "Prediction Pipeline",
    body: "At inference time we pre-compute the DenseNet201 feature once, then run the decoder in a greedy loop: tokenize the running caption → pad → predict next-word distribution → argmax → append word. The loop terminates on endseq or after max_length iterations.",
    points: [
      "Strategy: greedy argmax decoding",
      "Loop: up to max_length iterations",
      "Stop conditions: endseq token or max length",
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
            A faithful implementation of the
            <br />
            <span className="text-gradient">Flickr8K notebook</span>
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="mt-4 text-lg text-muted-foreground"
          >
            Every layer, every hyperparameter, every preprocessing step matches
            the original Kaggle notebook — only wrapped in a production-grade
            FastAPI service.
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
