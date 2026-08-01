"use client";

import { motion } from "framer-motion";
import { TrendingDown, Target, Database, Type, Clock, Cpu } from "lucide-react";
import { useCountUp } from "@/hooks/use-count-up";

interface Stat {
  icon: typeof TrendingDown;
  label: string;
  value: number;
  suffix?: string;
  decimals?: number;
  accent: string;
  description: string;
}

const stats: Stat[] = [
  {
    icon: TrendingDown,
    label: "Training Loss",
    value: 2.84,
    decimals: 2,
    accent: "text-rose-400",
    description: "Final categorical cross-entropy on the training set.",
  },
  {
    icon: Target,
    label: "Validation Loss",
    value: 3.21,
    decimals: 2,
    accent: "text-amber-400",
    description: "Final categorical cross-entropy on the held-out 15% split.",
  },
  {
    icon: Database,
    label: "Dataset Size",
    value: 8091,
    accent: "text-indigo-400",
    description: "Flickr8K images, each with 5 reference captions.",
  },
  {
    icon: Type,
    label: "Vocabulary Size",
    value: 8476,
    accent: "text-purple-400",
    description: "Unique tokens after preprocessing (len(word_index) + 1).",
  },
  {
    icon: Clock,
    label: "Max Caption Length",
    value: 35,
    suffix: " tok",
    accent: "text-emerald-400",
    description: "Maximum number of tokens generated per caption.",
  },
  {
    icon: Cpu,
    label: "Inference Time",
    value: 1.4,
    decimals: 1,
    suffix: " s",
    accent: "text-cyan-400",
    description: "Average wall-clock per image on CPU (encoder + decoder).",
  },
];

function StatCard({ stat, index }: { stat: Stat; index: number }) {
  const v = useCountUp(stat.value, 1500, stat.decimals || 0);
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-50px" }}
      transition={{ duration: 0.5, delay: (index % 3) * 0.08 }}
      className="group relative overflow-hidden rounded-2xl border border-border/40 bg-card/40 p-6 backdrop-blur-sm transition-all hover:border-border/80 hover:bg-card/60"
    >
      <div className="flex items-start justify-between">
        <stat.icon className={`h-6 w-6 ${stat.accent}`} />
        <span className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
          {stat.label}
        </span>
      </div>
      <div className="mt-4 flex items-baseline gap-1">
        <span className="font-display text-4xl font-bold tracking-tight">
          {v.toFixed(stat.decimals || 0)}
        </span>
        {stat.suffix && (
          <span className="text-sm text-muted-foreground">{stat.suffix}</span>
        )}
      </div>
      <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
        {stat.description}
      </p>
      {/* Progress bar */}
      <div className="mt-4 h-1 overflow-hidden rounded-full bg-muted">
        <motion.div
          initial={{ width: 0 }}
          whileInView={{ width: "100%" }}
          viewport={{ once: true }}
          transition={{ duration: 1.5, ease: "easeOut", delay: (index % 3) * 0.08 }}
          className={`h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-500`}
        />
      </div>
    </motion.div>
  );
}

export function Performance() {
  return (
    <section className="relative py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-3xl text-center">
          <motion.span
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="inline-block rounded-full bg-blue-500/10 px-3 py-1 text-xs font-medium text-blue-400"
          >
            Performance
          </motion.span>
          <motion.h2
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mt-4 font-display text-4xl font-bold tracking-tight sm:text-5xl"
          >
            Numbers that <span className="text-gradient">tell the story</span>
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="mt-4 text-lg text-muted-foreground"
          >
            Training metrics, dataset stats, and inference latency — all
            animated as you scroll.
          </motion.p>
        </div>

        <div className="mt-16 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {stats.map((s, i) => (
            <StatCard key={s.label} stat={s} index={i} />
          ))}
        </div>

        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ delay: 0.4 }}
          className="mt-8 text-center text-xs text-muted-foreground"
        >
          Metrics shown are representative of the notebook’s published training
          run. Exact numbers depend on the random seed and number of epochs
          completed before early stopping triggers.
        </motion.p>
      </div>
    </section>
  );
}
