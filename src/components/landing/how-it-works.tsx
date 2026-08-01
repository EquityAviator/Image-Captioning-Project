"use client";

import { motion } from "framer-motion";
import { Upload, Scan, Type, Eye } from "lucide-react";

const steps = [
  {
    n: "01",
    icon: Upload,
    title: "Upload image",
    description:
      "Drag-and-drop or click to upload any PNG, JPG, or JPEG image. Files up to 10 MB are supported and processed entirely on the server.",
  },
  {
    n: "02",
    icon: Scan,
    title: "Extract features",
    description:
      "The image is resized to 224×224 and passed through the DenseNet201 encoder, producing a 1920-dim feature vector that summarises the visual content.",
  },
  {
    n: "03",
    icon: Type,
    title: "Decode caption",
    description:
      "Starting from the special “startseq” token, the LSTM decoder samples one word at a time until it emits “endseq” or hits the max length.",
  },
  {
    n: "04",
    icon: Eye,
    title: "Display result",
    description:
      "The generated caption, inference time, and confidence score are returned to the UI with a beautiful animated reveal and a copy-to-clipboard button.",
  },
];

export function HowItWorks() {
  return (
    <section className="relative py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-3xl text-center">
          <motion.span
            initial={{ opacity: 0, y: 12 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="inline-block rounded-full bg-purple-500/10 px-3 py-1 text-xs font-medium text-purple-400"
          >
            How It Works
          </motion.span>
          <motion.h2
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="mt-4 font-display text-4xl font-bold tracking-tight sm:text-5xl"
          >
            Four steps from <span className="text-gradient-accent">upload to caption</span>
          </motion.h2>
        </div>

        <div className="mt-16 grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-4">
          {steps.map((s, i) => (
            <motion.div
              key={s.n}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              className="group relative"
            >
              <div className="relative h-full overflow-hidden rounded-2xl border border-border/40 bg-card/40 p-6 backdrop-blur-sm transition-all hover:border-indigo-500/40 hover:bg-card/60">
                <span className="pointer-events-none absolute -right-4 -top-8 font-display text-8xl font-bold text-indigo-500/5 transition-colors group-hover:text-indigo-500/10">
                  {s.n}
                </span>
                <div className="relative">
                  <div className="inline-flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 ring-1 ring-inset ring-indigo-500/30">
                    <s.icon className="h-6 w-6 text-indigo-400" />
                  </div>
                  <h3 className="mt-5 font-display text-lg font-semibold">
                    {s.title}
                  </h3>
                  <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                    {s.description}
                  </p>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
