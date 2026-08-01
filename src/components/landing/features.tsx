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
      "Single-image inference completes in under two seconds thanks to a pre-computed feature pipeline and a lightweight LSTM decoder.",
    accent: "from-amber-400 to-orange-500",
  },
  {
    icon: Brain,
    title: "Deep Learning Powered",
    description:
      "Trained end-to-end on the Flickr8K dataset with categorical cross-entropy and Adam optimiser, using teacher-forcing and early stopping.",
    accent: "from-indigo-400 to-purple-500",
  },
  {
    icon: Network,
    title: "DenseNet201 Encoder",
    description:
      "Uses the ImageNet-pretrained DenseNet201 backbone with the classification head removed, producing a 1920-dim feature vector per image.",
    accent: "from-blue-400 to-cyan-500",
  },
  {
    icon: Repeat,
    title: "LSTM Decoder",
    description:
      "An LSTM with 256 units decodes the image feature vector into a token sequence one word at a time, conditioned on the previous tokens.",
    accent: "from-emerald-400 to-teal-500",
  },
  {
    icon: Cpu,
    title: "TensorFlow Backend",
    description:
      "The FastAPI service loads the exact Keras model from the notebook — no rewrite, no additional training, just inference.",
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
