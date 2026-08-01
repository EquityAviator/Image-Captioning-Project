"use client";

import { Sparkles, Github, Linkedin, Mail, Heart } from "lucide-react";
import { useNav } from "@/lib/nav-store";

export function Footer() {
  const setView = useNav((s) => s.setView);
  const year = new Date().getFullYear();

  return (
    <footer className="mt-16 border-t border-border/40 bg-card/30 backdrop-blur-sm">
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-4">
          {/* Brand */}
          <div className="md:col-span-2">
            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 via-purple-500 to-blue-500 shadow-lg shadow-indigo-500/30">
                <Sparkles className="h-5 w-5 text-white" />
              </div>
              <div>
                <p className="font-display text-lg font-bold">CaptionAI</p>
                <p className="text-xs text-muted-foreground">
                  Deep Learning Image Captioning
                </p>
              </div>
            </div>
            <p className="mt-4 max-w-md text-sm leading-relaxed text-muted-foreground">
              An end-to-end AI web application that converts images into natural
              language descriptions using a DenseNet201 encoder and LSTM decoder
              trained on the Flickr8K dataset. Built with Next.js, FastAPI, and
              TensorFlow.
            </p>
            <div className="mt-5 flex items-center gap-3">
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="GitHub"
                className="flex h-9 w-9 items-center justify-center rounded-lg bg-muted/50 transition-colors hover:bg-muted hover:text-indigo-400"
              >
                <Github className="h-4 w-4" />
              </a>
              <a
                href="https://linkedin.com"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="LinkedIn"
                className="flex h-9 w-9 items-center justify-center rounded-lg bg-muted/50 transition-colors hover:bg-muted hover:text-indigo-400"
              >
                <Linkedin className="h-4 w-4" />
              </a>
              <a
                href="mailto:hello@captionai.dev"
                aria-label="Email"
                className="flex h-9 w-9 items-center justify-center rounded-lg bg-muted/50 transition-colors hover:bg-muted hover:text-indigo-400"
              >
                <Mail className="h-4 w-4" />
              </a>
            </div>
          </div>

          {/* Product */}
          <div>
            <h4 className="font-display text-sm font-semibold uppercase tracking-wider text-foreground">
              Product
            </h4>
            <ul className="mt-4 space-y-2.5 text-sm">
              <li>
                <button
                  onClick={() => setView("dashboard")}
                  className="text-muted-foreground transition-colors hover:text-foreground"
                >
                  Dashboard
                </button>
              </li>
              <li>
                <button
                  onClick={() => setView("model-info")}
                  className="text-muted-foreground transition-colors hover:text-foreground"
                >
                  Model Info
                </button>
              </li>
              <li>
                <button
                  onClick={() => setView("settings")}
                  className="text-muted-foreground transition-colors hover:text-foreground"
                >
                  Settings
                </button>
              </li>
              <li>
                <button
                  onClick={() => setView("history")}
                  className="text-muted-foreground transition-colors hover:text-foreground"
                >
                  History
                </button>
              </li>
            </ul>
          </div>

          {/* About */}
          <div>
            <h4 className="font-display text-sm font-semibold uppercase tracking-wider text-foreground">
              About
            </h4>
            <ul className="mt-4 space-y-2.5 text-sm">
              <li>
                <a
                  href="https://www.kaggle.com/datasets/adityajn105/flickr8k"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted-foreground transition-colors hover:text-foreground"
                >
                  Flickr8K Dataset
                </a>
              </li>
              <li>
                <a
                  href="https://keras.io/api/applications/densenet/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted-foreground transition-colors hover:text-foreground"
                >
                  DenseNet201
                </a>
              </li>
              <li>
                <a
                  href="https://www.tensorflow.org/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted-foreground transition-colors hover:text-foreground"
                >
                  TensorFlow
                </a>
              </li>
              <li>
                <a
                  href="https://nextjs.org/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-muted-foreground transition-colors hover:text-foreground"
                >
                  Next.js
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-10 flex flex-col items-center justify-between gap-4 border-t border-border/40 pt-6 sm:flex-row">
          <p className="text-xs text-muted-foreground">
            © {year} CaptionAI. All rights reserved.
          </p>
          <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
            Built with <Heart className="h-3 w-3 fill-rose-500 text-rose-500" /> using
            TensorFlow & Next.js
          </p>
        </div>
      </div>
    </footer>
  );
}
