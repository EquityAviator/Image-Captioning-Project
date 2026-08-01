"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
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
  Clock,
  Box,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { apiClient, ApiError } from "@/lib/api";
import type { ModelInfoResponse } from "@/lib/types";

const sections = [
  {
    icon: Network,
    title: "DenseNet201",
    body: "Densely Connected Convolutional Network with 201 layers, pretrained on ImageNet (1.28M images, 1000 classes). Each layer receives feature maps from all preceding layers, enabling strong gradient flow and feature reuse. We strip the classification head and apply global average pooling.",
  },
  {
    icon: Layers,
    title: "LSTM Decoder",
    body: "Long Short-Term Memory network with 256 hidden units. Recurrently generates one word at a time, conditioned on the image feature vector and the previously generated tokens. The LSTM cell uses the standard gating mechanism (input, forget, output gates) to selectively remember and forget information across timesteps.",
  },
  {
    icon: Type,
    title: "Embedding Layer",
    body: "A learned 256-dimensional word embedding. Maps each token in the caption vocabulary to a dense vector. The embedding is trained jointly with the rest of the decoder so that semantically similar words end up close together in the embedding space.",
  },
  {
    icon: Hash,
    title: "Tokenizer & Vocabulary",
    body: "Built with Keras’ Tokenizer on the training captions after lowercasing, removing non-alphabetic characters, and wrapping with startseq / endseq special tokens. Each unique word is assigned a unique integer index; the vocabulary size is len(word_index) + 1.",
  },
  {
    icon: Database,
    title: "Training Dataset",
    body: "Flickr8K — a collection of 8,091 Flickr photographs, each annotated with 5 crowd-sourced captions (40,455 captions total). The dataset is small enough to train on a single GPU yet diverse enough to teach general visual concepts.",
  },
  {
    icon: GitBranch,
    title: "Prediction Flow",
    body: "Greedy decoding: starting from the special startseq token, we run the decoder one step at a time. At each step we argmax over the vocabulary softmax, append the chosen word to the running caption, and re-tokenise the new sequence. The loop terminates when the endseq token is emitted or the max caption length is reached.",
  },
];

export function ModelInfoView() {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    apiClient
      .getModelInfo()
      .then(setInfo)
      .catch((e) => {
        if (e instanceof ApiError) setError(e.detail);
        else setError(String(e));
      });
  }, []);

  return (
    <div className="space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="font-display text-3xl font-bold tracking-tight">
          Model Info
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Architecture, training metadata, and live backend status.
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
          icon={Cpu}
          label="Provider"
          value={info?.provider || "—"}
          tone="indigo"
        />
        <StatusCard
          icon={Box}
          label="Weights"
          value={info ? (info.weights_loaded ? "Loaded" : "Fallback") : "—"}
          tone={info?.weights_loaded ? "emerald" : "amber"}
        />
        <StatusCard
          icon={Activity}
          label="TensorFlow"
          value={info?.tensorflow_version || "—"}
          tone="purple"
        />
      </div>

      {/* Architecture diagram */}
      <Card className="overflow-hidden border-border/40 bg-card/40 p-6 backdrop-blur-sm">
        <h2 className="font-display text-xl font-bold">Architecture Pipeline</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          End-to-end data flow from raw pixels to a generated caption.
        </p>

        <div className="mt-6 space-y-3">
          <PipelineRow
            n={1}
            label="Input image"
            sub="224 × 224 × 3 (RGB)"
            color="from-blue-400 to-cyan-500"
          />
          <PipelineArrow label="resize + normalise" />
          <PipelineRow
            n={2}
            label="DenseNet201 (frozen)"
            sub="ImageNet-pretrained encoder"
            color="from-indigo-400 to-purple-500"
          />
          <PipelineArrow label="global avg pool" />
          <PipelineRow
            n={3}
            label="Feature vector"
            sub="1920-dim"
            color="from-purple-400 to-fuchsia-500"
          />
          <PipelineArrow label="Dense(256) + Reshape(1, 256)" />
          <PipelineRow
            n={4}
            label="Concatenate with Embedding(256)"
            sub="image feats + caption tokens"
            color="from-fuchsia-400 to-pink-500"
          />
          <PipelineArrow label="LSTM(256) + Dropout + Add" />
          <PipelineRow
            n={5}
            label="Dense(128) + Dropout"
            sub="regularised features"
            color="from-pink-400 to-rose-500"
          />
          <PipelineArrow label="Dense(vocab_size, softmax)" />
          <PipelineRow
            n={6}
            label="Next-word distribution"
            sub="argmax → token → append"
            color="from-amber-400 to-orange-500"
          />
        </div>
      </Card>

      {/* Hyperparameters grid */}
      {info && (
        <Card className="border-border/40 bg-card/40 p-6 backdrop-blur-sm">
          <h2 className="font-display text-xl font-bold">Hyperparameters</h2>
          <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
            <Stat label="Vocabulary size" value={info.vocab_size.toLocaleString()} />
            <Stat label="Max caption length" value={`${info.max_length} tokens`} />
            <Stat label="Embedding dim" value={`${info.embed_dim}`} />
            <Stat label="LSTM units" value={`${info.lstm_units}`} />
            <Stat label="Dense units" value={`${info.dense_units}`} />
            <Stat label="Dropout" value={info.dropout.toString()} />
            <Stat label="Image size" value={`${info.image_size}px`} />
            <Stat label="Encoder dim" value={`${info.encoder_feature_dim}`} />
          </div>
        </Card>
      )}

      {/* Sections */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {sections.map((s, i) => (
          <motion.div
            key={s.title}
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4, delay: (i % 2) * 0.05 }}
          >
            <Card className="h-full border-border/40 bg-card/40 p-5 backdrop-blur-sm">
              <div className="flex items-start gap-3">
                <div className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500/20 to-purple-500/20 ring-1 ring-inset ring-indigo-500/30">
                  <s.icon className="h-5 w-5 text-indigo-400" />
                </div>
                <div>
                  <h3 className="font-display font-semibold">{s.title}</h3>
                  <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
                    {s.body}
                  </p>
                </div>
              </div>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Training metadata */}
      {info?.training_metadata &&
        Object.keys(info.training_metadata).length > 0 && (
          <Card className="border-border/40 bg-card/40 p-6 backdrop-blur-sm">
            <h2 className="font-display text-xl font-bold">
              Training Metadata (from metadata.json)
            </h2>
            <pre className="mt-4 overflow-x-auto rounded-lg border border-border/40 bg-muted/30 p-4 font-mono text-xs">
              {JSON.stringify(info.training_metadata, null, 2)}
            </pre>
          </Card>
        )}
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
  const tones: Record<typeof tone, string> = {
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
