"use client";

import { motion } from "framer-motion";
import { ArrowRight, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useNav } from "@/lib/nav-store";

export function DemoCTA() {
  const setView = useNav((s) => s.setView);
  return (
    <section className="relative py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="relative overflow-hidden rounded-3xl border border-border/40 bg-gradient-to-br from-indigo-500/10 via-purple-500/10 to-blue-500/10 p-10 backdrop-blur-sm sm:p-16"
        >
          {/* Decorative blobs */}
          <div className="absolute -left-20 -top-20 h-72 w-72 rounded-full bg-indigo-500/20 blur-3xl" />
          <div className="absolute -bottom-20 -right-20 h-72 w-72 rounded-full bg-purple-500/20 blur-3xl" />

          <div className="relative mx-auto max-w-2xl text-center">
            <div className="inline-flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-500 shadow-xl shadow-indigo-500/30">
              <Sparkles className="h-7 w-7 text-white" />
            </div>
            <h2 className="mt-6 font-display text-4xl font-bold tracking-tight sm:text-5xl">
              Ready to caption your
              <br />
              <span className="text-gradient">first image?</span>
            </h2>
            <p className="mt-4 text-lg text-muted-foreground">
              Upload any photo and watch the model generate a natural language
              description in real time. No signup, no API key, no waiting.
            </p>
            <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
              <Button
                size="lg"
                onClick={() => setView("dashboard")}
                className="group h-12 px-8 bg-gradient-to-r from-indigo-500 via-purple-500 to-blue-500 text-white shadow-lg shadow-indigo-500/30"
              >
                Launch Dashboard
                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
              </Button>
              <Button
                size="lg"
                variant="outline"
                onClick={() => setView("model-info")}
                className="h-12 px-8 border-border/60 bg-card/40 backdrop-blur-md"
              >
                Explore the Model
              </Button>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
