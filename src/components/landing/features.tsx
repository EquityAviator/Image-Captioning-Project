"use client";

import { motion } from "framer-motion";
import {
  Zap,
  Brain,
  Network,
  Repeat,
  Cpu,
  Smartphone,
  type LucideIcon,
} from "lucide-react";
import { Card } from "@/components/ui/card";

interface Feature {
  icon: LucideIcon;
  title: string;
  description: string;
  accent: string;
}

const features: Feature[] = [
  {
    icon: Zap,
    title: "Fast Prediction",
    description:
      "Warm in-domain captions in ~590 ms — a 53× serving win from incremental state-carrying beam search, fp16 features, and a feature cache.",
    accent: "from-amber-400 to-orange-500",
  },
  {
    icon: Brain,
    title: "GRPO-Reinforced Decoder",
    description:
      "Cross-entropy pretraining, then Group Relative Policy Optimization against a CIDEr-D reward — BLEU-1 0.6559, word precision 70.7%.",
    accent: "from-indigo-400 to-purple-500",
  },
  {
    icon: Network,
    title: "CLIP ViT-B/16 Features",
    description:
      "A frozen CLIP vision trunk (400M image-text pairs) supplies 196 patch tokens per image — language-aligned features that beat ImageNet backbones on every metric.",
    accent: "from-blue-400 to-cyan-500",
  },
  {
    icon: Repeat,
    title: "Bahdanau Attention",
    description:
      "An additive attention head scores every image patch against the LSTM(512) state at each step, so the model learns WHERE to look for each word.",
    accent: "from-emerald-400 to-teal-500",
  },
  {
    icon: Cpu,
    title: "Hybrid OOD Routing",
    description:
      "The same CLIP trunk zero-shot classifies each image: real photos go to the specialist, cartoons/screenshots auto-route to BLIP — badged in the UI.",
    accent: "from-rose-400 to-pink-500",
  },
  {
    icon: Smartphone,
    title: "Fully Responsive",
    description:
      "Built mobile-first with Tailwind CSS 4 and shadcn/ui. Works beautifully across phones, tablets, and large desktop displays.",
    accent: "from-violet-400 to-fuchsia-500",
  },
];

export function Features() {
  return (
    <section className="relative py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-3xl text-center">
          <motion.h2
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
            className="font-display text-4xl font-bold tracking-tight sm:text-5xl"
          >
            Everything you need for
            <br />
            <span className="text-gradient-accent">production-grade captioning</span>
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="mt-4 text-lg text-muted-foreground"
          >
            A complete deep-learning pipeline exposed through a modern,
            accessible, animated web interface.
          </motion.p>
        </div>

        <div className="mt-16 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.5, delay: (i % 3) * 0.08 }}
            >
              <Card className="group relative h-full overflow-hidden border-border/40 bg-card/40 p-6 backdrop-blur-sm transition-all hover:border-border/80 hover:bg-card/60">
                <div
                  className={`absolute inset-x-0 -top-px h-px bg-gradient-to-r ${f.accent} opacity-0 transition-opacity group-hover:opacity-100`}
                />
                <div
                  className={`inline-flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br ${f.accent} shadow-lg`}
                >
                  <f.icon className="h-6 w-6 text-white" />
                </div>
                <h3 className="mt-5 font-display text-lg font-semibold">
                  {f.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                  {f.description}
                </p>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
