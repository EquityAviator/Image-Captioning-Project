"use client";

import { motion } from "framer-motion";
import { Image as ImageIcon, ArrowDown, Brain, Repeat, Type } from "lucide-react";

const steps = [
  {
    icon: ImageIcon,
    label: "Image",
    sub: "PNG / JPG / JPEG",
    color: "from-blue-400 to-cyan-500",
  },
  {
    icon: Brain,
    label: "CLIP ViT-B/16",
    sub: "Frozen vision trunk",
    color: "from-indigo-400 to-purple-500",
  },
  {
    icon: Type,
    label: "Patch Features",
    sub: "196 × 768-dim",
    color: "from-purple-400 to-fuchsia-500",
  },
  {
    icon: Repeat,
    label: "Attention LSTM",
    sub: "Bahdanau + GRPO",
    color: "from-emerald-400 to-teal-500",
  },
  {
    icon: Type,
    label: "Generated Caption",
    sub: "Natural language",
    color: "from-amber-400 to-orange-500",
  },
];

export function Architecture() {
  return (
    <section className="relative py-24">
      <div className="absolute inset-0 -z-10 dot-pattern opacity-30" />
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-3xl text-center">
          <motion.span
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="inline-block rounded-full bg-indigo-500/10 px-3 py-1 text-xs font-medium text-indigo-400"
          >
            Architecture
          </motion.span>
          <motion.h2
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mt-4 font-display text-4xl font-bold tracking-tight sm:text-5xl"
          >
            From pixels to <span className="text-gradient">captions</span>
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="mt-4 text-lg text-muted-foreground"
          >
            A two-stage encoder-decoder pipeline that converts raw RGB pixels
            into fluent natural language descriptions.
          </motion.p>
        </div>

        {/* Pipeline */}
        <div className="mt-16">
          <div className="flex flex-col items-stretch gap-4 lg:flex-row lg:items-center lg:justify-between">
            {steps.map((s, i) => (
              <div key={s.label} className="flex flex-1 items-center gap-4 lg:flex-col">
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: i * 0.15 }}
                  className="relative flex flex-1 flex-col items-center lg:flex-none"
                >
                  <div
                    className={`group relative flex h-24 w-24 items-center justify-center rounded-2xl bg-gradient-to-br ${s.color} shadow-xl transition-transform hover:scale-105 sm:h-28 sm:w-28`}
                  >
                    <div className="absolute inset-0 rounded-2xl bg-white/10 backdrop-blur-sm" />
                    <s.icon className="relative h-10 w-10 text-white sm:h-12 sm:w-12" />
                  </div>
                  <div className="mt-4 text-center">
                    <p className="font-display font-semibold">{s.label}</p>
                    <p className="text-xs text-muted-foreground">{s.sub}</p>
                  </div>
                </motion.div>

                {/* Arrow */}
                {i < steps.length - 1 && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.15 + 0.2 }}
                    className="flex items-center justify-center lg:rotate-0 rotate-90"
                  >
                    <motion.div
                      animate={{ x: [0, 6, 0] }}
                      transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                    >
                      <ArrowDown className="h-5 w-5 text-muted-foreground lg:rotate-[-90deg]" />
                    </motion.div>
                  </motion.div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Bottom callout */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="mx-auto mt-16 max-w-3xl"
        >
          <div className="rounded-2xl border border-border/40 bg-card/40 p-6 backdrop-blur-sm">
            <p className="text-center text-sm leading-relaxed text-muted-foreground">
              <span className="font-semibold text-foreground">Encoder:</span>{" "}
              A frozen CLIP ViT-B/16 vision trunk (pretrained on 400M
              image-text pairs) turns each 224×224×3 image into 196 patch
              tokens of 768 dims — spatial detail that global average pooling
              would erase.{" "}
              <span className="font-semibold text-foreground">Decoder:</span>{" "}
              A Bahdanau additive attention head scores every patch against
              the LSTM(512) state at each step, so the model learns WHERE to
              look for each word. Weights are fine-tuned with GRPO against a
              CIDEr-D reward, then decoded with beam-5 search and a min-length
              guard.
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
