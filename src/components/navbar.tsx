"use client";

import { Sparkles, Github, Linkedin, Mail } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";
import { useNav } from "@/lib/nav-store";
import { motion } from "framer-motion";

export function Navbar() {
  const setView = useNav((s) => s.setView);

  return (
    <motion.header
      initial={{ y: -24, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      className="sticky top-0 z-40 w-full"
    >
      <div className="glass-strong border-b border-border/40">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          {/* Logo */}
          <button
            onClick={() => setView("landing")}
            className="group flex items-center gap-2.5"
            aria-label="CaptionAI home"
          >
            <div className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 via-purple-500 to-blue-500 shadow-lg shadow-indigo-500/30 transition-transform group-hover:scale-110">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div className="hidden flex-col items-start sm:flex">
              <span className="font-display text-base font-bold leading-none tracking-tight">
                CaptionAI
              </span>
              <span className="text-[10px] uppercase tracking-widest text-muted-foreground">
                Image Captioning
              </span>
            </div>
          </button>

          {/* Right side */}
          <div className="flex items-center gap-1 sm:gap-2">
            <nav className="hidden items-center gap-1 md:flex">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setView("model-info")}
                className="text-muted-foreground hover:text-foreground"
              >
                Model
              </Button>
              <Button
                variant="ghost"
                size="sm"
                asChild
                className="text-muted-foreground hover:text-foreground"
              >
                <a
                  href="https://github.com"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Github className="mr-1.5 h-4 w-4" /> Source
                </a>
              </Button>
            </nav>

            <Button
              variant="ghost"
              size="icon"
              asChild
              className="rounded-full"
              aria-label="GitHub"
            >
              <a href="https://github.com" target="_blank" rel="noopener noreferrer">
                <Github className="h-4 w-4" />
              </a>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              asChild
              className="hidden rounded-full sm:inline-flex"
              aria-label="LinkedIn"
            >
              <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer">
                <Linkedin className="h-4 w-4" />
              </a>
            </Button>
            <Button
              variant="ghost"
              size="icon"
              asChild
              className="hidden rounded-full sm:inline-flex"
              aria-label="Contact"
            >
              <a href="mailto:hello@captionai.dev">
                <Mail className="h-4 w-4" />
              </a>
            </Button>
            <ThemeToggle />

            <Button
              size="sm"
              onClick={() => setView("dashboard")}
              className="ml-1 bg-gradient-to-r from-indigo-500 to-purple-500 text-white hover:from-indigo-600 hover:to-purple-600"
            >
              Try Now
            </Button>
          </div>
        </div>
      </div>
    </motion.header>
  );
}
