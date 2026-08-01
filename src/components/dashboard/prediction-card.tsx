"use client";

import { motion, AnimatePresence } from "framer-motion";
import {
  Copy,
  Check,
  Download,
  Share2,
  RefreshCw,
  Volume2,
  Square,
  Sparkles,
  Clock,
  TrendingUp,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useState, useEffect } from "react";
import { toast } from "sonner";
import { useTypingEffect } from "@/hooks/use-typing-effect";
import { useSpeechSynthesis as useSpeech } from "@/hooks/use-speech";
import { useSettings } from "@/lib/settings-store";
import type { PredictResponse } from "@/lib/types";

interface Props {
  result: PredictResponse;
  imageUrl: string | null;
  onRegenerate: () => void;
}

export function PredictionCard({ result, imageUrl, onRegenerate }: Props) {
  const [copied, setCopied] = useState(false);
  const typingEffect = useSettings((s) => s.typingEffectEnabled);
  const speechSynth = useSettings((s) => s.speechSynthesisEnabled);
  const speech = useSpeech();

  const caption = useTypingEffect(
    result.caption,
    typingEffect && speechSynth === false ? true : true,
    16,
  );

  useEffect(() => {
    if (speechSynth && result.caption && typingEffect === false) {
      // auto-speak disabled by default; explicit button below
    }
  }, [speechSynth, result.caption, typingEffect]);

  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(result.caption);
      setCopied(true);
      toast.success("Caption copied to clipboard");
      setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error("Couldn’t copy to clipboard");
    }
  };

  const onDownload = () => {
    const text = `CaptionAI — Generated Caption
===================================
Caption:         ${result.caption}
Inference time:  ${result.inference_time}
Confidence:      ${(result.confidence * 100).toFixed(1)}%
Provider:        ${result.provider}
Timestamp:       ${new Date().toISOString()}

Image: ${imageUrl || "(none)"}
`;
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `caption-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Result downloaded");
  };

  const onShare = async () => {
    const shareData = {
      title: "CaptionAI",
      text: result.caption,
    };
    if (navigator.share) {
      try {
        await navigator.share(shareData);
      } catch {
        /* user cancelled */
      }
    } else {
      await navigator.clipboard.writeText(result.caption);
      toast.success("Share not supported — caption copied instead");
    }
  };

  const onSpeak = () => {
    if (speech.speaking) {
      speech.cancel();
    } else {
      speech.speak(result.caption);
    }
  };

  const confidencePct = Math.round(result.confidence * 100);

  return (
    <motion.div
      initial={{ opacity: 0, y: 16, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="relative overflow-hidden rounded-2xl border border-border/40 bg-gradient-to-br from-indigo-500/5 via-purple-500/5 to-blue-500/5 p-6 backdrop-blur-xl"
    >
      {/* Glow accent */}
      <div className="pointer-events-none absolute -right-12 -top-12 h-40 w-40 rounded-full bg-indigo-500/20 blur-3xl" />

      <div className="relative flex items-start justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-purple-500">
            <Sparkles className="h-4 w-4 text-white" />
          </div>
          <span className="font-display text-sm font-semibold">
            Generated Caption
          </span>
        </div>
        <Badge
          variant="outline"
          className="border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
        >
          {result.provider === "huggingface-blip" ? "BLIP" : "Notebook"}
        </Badge>
      </div>

      {/* Caption */}
      <div className="relative mt-5 min-h-[5rem]">
        <p className="font-display text-xl font-medium leading-relaxed tracking-tight sm:text-2xl">
          <span className="typing-cursor">{caption}</span>
        </p>
      </div>

      {/* Stats */}
      <div className="relative mt-5 grid grid-cols-2 gap-3">
        <div className="rounded-xl border border-border/40 bg-card/40 p-3">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Clock className="h-3 w-3" />
            Inference time
          </div>
          <p className="mt-1 font-mono text-sm font-semibold">
            {result.inference_time}
          </p>
        </div>
        <div className="rounded-xl border border-border/40 bg-card/40 p-3">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <TrendingUp className="h-3 w-3" />
            Confidence
          </div>
          <div className="mt-1 flex items-center gap-2">
            <p className="font-mono text-sm font-semibold">{confidencePct}%</p>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${confidencePct}%` }}
                transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
                className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-teal-500"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="relative mt-5 flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          onClick={onCopy}
          className="h-9 bg-gradient-to-r from-indigo-500 to-purple-500 text-white hover:from-indigo-600 hover:to-purple-600"
        >
          <AnimatePresence mode="wait" initial={false}>
            {copied ? (
              <motion.span
                key="copied"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="flex items-center"
              >
                <Check className="mr-1.5 h-3.5 w-3.5" /> Copied
              </motion.span>
            ) : (
              <motion.span
                key="copy"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                className="flex items-center"
              >
                <Copy className="mr-1.5 h-3.5 w-3.5" /> Copy
              </motion.span>
            )}
          </AnimatePresence>
        </Button>
        <Button size="sm" variant="outline" onClick={onDownload} className="h-9">
          <Download className="mr-1.5 h-3.5 w-3.5" /> Download
        </Button>
        <Button size="sm" variant="outline" onClick={onShare} className="h-9">
          <Share2 className="mr-1.5 h-3.5 w-3.5" /> Share
        </Button>
        {speech.supported && (
          <Button
            size="sm"
            variant="outline"
            onClick={onSpeak}
            className="h-9"
          >
            {speech.speaking ? (
              <>
                <Square className="mr-1.5 h-3.5 w-3.5" /> Stop
              </>
            ) : (
              <>
                <Volume2 className="mr-1.5 h-3.5 w-3.5" /> Speak
              </>
            )}
          </Button>
        )}
        <Button
          size="sm"
          variant="ghost"
          onClick={onRegenerate}
          className="h-9 ml-auto"
        >
          <RefreshCw className="mr-1.5 h-3.5 w-3.5" /> Regenerate
        </Button>
      </div>
    </motion.div>
  );
}
