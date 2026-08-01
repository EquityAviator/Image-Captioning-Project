"use client";

import { motion } from "framer-motion";
import { Sparkles, ArrowRight, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useNav } from "@/lib/nav-store";

export function Hero() {
  const setView = useNav((s) => s.setView);

  return (
    <section className="relative overflow-hidden">
      {/* Animated gradient background */}
      <div className="absolute inset-0 -z-10">
        <div className="animated-gradient-bg absolute inset-0 opacity-40" />
        <div className="absolute inset-0 grid-pattern opacity-50" />
        {/* Floating blobs */}
        <div className="absolute -left-32 top-10 h-96 w-96 rounded-full bg-indigo-500/30 blur-3xl animate-blob" />
        <div className="absolute -right-20 top-32 h-80 w-80 rounded-full bg-purple-500/30 blur-3xl animate-blob-slow" />
        <div className="absolute bottom-0 left-1/3 h-72 w-72 rounded-full bg-blue-500/20 blur-3xl animate-blob" />
        <div className="absolute right-1/4 top-1/2 h-64 w-64 rounded-full bg-emerald-500/20 blur-3xl animate-blob-slow" />
      </div>

      <div className="mx-auto max-w-7xl px-4 pb-20 pt-24 sm:px-6 sm:pt-32 lg:px-8 lg:pt-40">
        <div className="mx-auto max-w-4xl text-center">
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="mb-6 inline-flex items-center gap-2 rounded-full border border-border/60 bg-card/40 px-4 py-1.5 text-xs font-medium text-muted-foreground backdrop-blur-md"
          >
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-500" />
            </span>
            <span>Powered by DenseNet201 + LSTM • TensorFlow Backend</span>
          </motion.div>

          {/* Title */}
          <motion.h1
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.1 }}
            className="font-display text-5xl font-bold tracking-tight sm:text-6xl lg:text-7xl"
          >
            <span className="text-gradient">AI Image Caption</span>
            <br />
            <span className="text-foreground">Generator</span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2 }}
            className="mx-auto mt-6 max-w-2xl text-lg text-muted-foreground sm:text-xl"
          >
            Generate natural language descriptions for any image using Deep
            Learning. A production-ready pipeline combining a DenseNet201
            encoder, an LSTM decoder, and the Flickr8K dataset.
          </motion.p>

          {/* CTAs */}
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
            className="mt-10 flex flex-col items-center justify-center gap-3 sm:flex-row"
          >
            <Button
              size="lg"
              onClick={() => setView("dashboard")}
              className="group h-12 px-8 bg-gradient-to-r from-indigo-500 via-purple-500 to-blue-500 text-white shadow-lg shadow-indigo-500/30 transition-all hover:shadow-xl hover:shadow-indigo-500/40"
            >
              <Sparkles className="mr-2 h-4 w-4 transition-transform group-hover:rotate-12" />
              Try Now
              <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
            </Button>
            <Button
              size="lg"
              variant="outline"
              onClick={() => setView("model-info")}
              className="h-12 px-8 border-border/60 bg-card/40 backdrop-blur-md"
            >
              <Play className="mr-2 h-4 w-4" />
              How It Works
            </Button>
          </motion.div>

          {/* Tech chips */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.7, delay: 0.5 }}
            className="mt-12 flex flex-wrap items-center justify-center gap-2 text-xs text-muted-foreground"
          >
            {[
              "Next.js 16",
              "FastAPI",
              "TensorFlow 2.21",
              "DenseNet201",
              "LSTM",
              "Flickr8K",
              "Tailwind CSS 4",
              "shadcn/ui",
            ].map((t) => (
              <span
                key={t}
                className="rounded-full border border-border/60 bg-card/40 px-3 py-1 backdrop-blur-md"
              >
                {t}
              </span>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
