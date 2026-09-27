"use client";

import { motion } from "framer-motion";
import {
  Moon,
  Sun,
  Monitor,
  Sparkles,
  History as HistoryIcon,
  Type,
  Volume2,
  Link2,
  Copy,
  Check,
  RefreshCw,
  Brain,
  Cpu,
  Eye,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { useSettings } from "@/lib/settings-store";
import { useTheme } from "next-themes";
import { useNav } from "@/lib/nav-store";
import { toast } from "sonner";
import { useState } from "react";
import type { ThemeMode, InferenceProvider } from "@/lib/types";

const themes: { key: ThemeMode; label: string; icon: typeof Moon }[] = [
  { key: "dark", label: "Dark", icon: Moon },
  { key: "light", label: "Light", icon: Sun },
  { key: "system", label: "System", icon: Monitor },
];

const providers: { key: InferenceProvider; label: string; description: string; icon: typeof Brain }[] = [
  { key: "auto", label: "Auto (trained CLIP+GRPO)", description: "Best model — CLIP ViT-B/16 + GRPO attention decoder (recommended)", icon: Brain },
  { key: "attention", label: "Attention v2 (CLIP + GRPO)", description: "Explicitly pin the trained PyTorch attention decoder", icon: Eye as unknown as typeof Brain },
  { key: "notebook-tensorflow", label: "Notebook (TensorFlow)", description: "Legacy Gen-1 DenseNet201 + LSTM — much weaker, for comparison only", icon: Cpu },
  { key: "huggingface-blip", label: "HuggingFace BLIP", description: "Pre-trained BLIP base model (works out of the box)", icon: Cpu },
];

export function SettingsView() {
  const s = useSettings();
  const { theme, setTheme } = useTheme();
  const setView = useNav((s) => s.setView);
  const [copied, setCopied] = useState(false);

  const copyApiUrl = () => {
    const url = s.apiBaseUrl || "/predict?XTransformPort=8000";
    navigator.clipboard.writeText(url);
    setCopied(true);
    toast.success("API URL copied");
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className="space-y-8">
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="font-display text-3xl font-bold tracking-tight">Settings</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Configure appearance, behaviour, and backend connection.
        </p>
      </motion.div>

      {/* Theme */}
      <Card className="border-border/40 bg-card/40 p-6 backdrop-blur-sm">
        <SectionHeader icon={Sun} title="Theme" description="Choose how CaptionAI looks." />
        <div className="mt-4 grid grid-cols-3 gap-3">
          {themes.map((t) => {
            const active = (theme || "dark") === t.key;
            return (
              <button
                key={t.key}
                onClick={() => {
                  setTheme(t.key);
                  s.setTheme(t.key);
                }}
                className={`group relative flex flex-col items-center gap-2 rounded-xl border-2 p-4 transition-all ${
                  active
                    ? "border-indigo-500 bg-indigo-500/10"
                    : "border-border/40 bg-card/30 hover:border-border"
                }`}
              >
                <t.icon
                  className={`h-6 w-6 ${active ? "text-indigo-400" : "text-muted-foreground"}`}
                />
                <span
                  className={`text-sm font-medium ${active ? "text-indigo-400" : ""}`}
                >
                  {t.label}
                </span>
              </button>
            );
          })}
        </div>
      </Card>

      {/* Behaviour */}
      <Card className="border-border/40 bg-card/40 p-6 backdrop-blur-sm">
        <SectionHeader
          icon={Sparkles}
          title="Behaviour"
          description="Control animations and prediction UX."
        />
        <div className="mt-4 space-y-1">
          <ToggleRow
            icon={Sparkles}
            label="Animations"
            description="Page transitions, hover effects, and micro-interactions."
            checked={s.animationsEnabled}
            onCheckedChange={(v) => s.set("animationsEnabled", v)}
          />
          <ToggleRow
            icon={HistoryIcon}
            label="Save to history"
            description="Store previous predictions in localStorage (last 50)."
            checked={s.historyEnabled}
            onCheckedChange={(v) => s.set("historyEnabled", v)}
          />
          <ToggleRow
            icon={Type}
            label="Typing effect"
            description="Reveal captions character-by-character for a typewriter feel."
            checked={s.typingEffectEnabled}
            onCheckedChange={(v) => s.set("typingEffectEnabled", v)}
          />
          <ToggleRow
            icon={Volume2}
            label="Speech synthesis"
            description="Enable the “Speak” button on prediction cards (uses browser TTS)."
            checked={s.speechSynthesisEnabled}
            onCheckedChange={(v) => s.set("speechSynthesisEnabled", v)}
          />
          <ToggleRow
            icon={Check}
            label="Auto-copy caption"
            description="Automatically copy each new caption to your clipboard."
            checked={s.autoCopyEnabled}
            onCheckedChange={(v) => s.set("autoCopyEnabled", v)}
          />
        </div>
      </Card>

      {/* Inference Provider */}
      <Card className="border-border/40 bg-card/40 p-6 backdrop-blur-sm">
        <SectionHeader
          icon={Brain}
          title="Inference Provider"
          description="Choose which model backend to use for caption generation."
        />
        <div className="mt-4 space-y-3">
          {providers.map((p) => {
            const active = s.inferenceProvider === p.key;
            return (
              <button
                key={p.key}
                onClick={() => s.setInferenceProvider(p.key)}
                className={`group relative flex items-center gap-4 rounded-xl border-2 p-4 transition-all ${
                  active
                    ? "border-indigo-500 bg-indigo-500/10"
                    : "border-border/40 bg-card/30 hover:border-border"
                }`}
              >
                <p.icon
                  className={`h-6 w-6 shrink-0 ${active ? "text-indigo-400" : "text-muted-foreground"}`}
                />
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span
                      className={`font-medium ${active ? "text-indigo-400" : ""}`}
                    >
                      {p.label}
                    </span>
                    {active && (
                      <span className="text-xs font-medium text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded">
                        Active
                      </span>
                    )}
                  </div>
                  <p className="mt-1 text-sm text-muted-foreground">{p.description}</p>
                </div>
                {active && (
                  <motion.div
                    layoutId="provider-active"
                    className="absolute inset-0 rounded-xl bg-indigo-500/10 ring-1 ring-inset ring-indigo-500/30"
                  />
                )}
              </button>
            );
          })}
        </div>
      </Card>

      {/* API connection */}
      <Card className="border-border/40 bg-card/40 p-6 backdrop-blur-sm">
        <SectionHeader
          icon={Link2}
          title="API URL"
          description="Optional. Leave blank to use the built-in gateway (recommended)."
        />
        <div className="mt-4 flex items-center gap-2">
          <Input
            value={s.apiBaseUrl}
            onChange={(e) => s.set("apiBaseUrl", e.target.value)}
            placeholder="https://your-backend.example.com"
            className="font-mono text-sm"
          />
          <Button variant="outline" onClick={copyApiUrl} className="shrink-0">
            {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          </Button>
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          Default: <code className="rounded bg-muted/30 px-1.5 py-0.5 font-mono">/predict?XTransformPort=8000</code>
        </p>
      </Card>

      {/* Reset */}
      <div className="flex justify-end">
        <Button
          variant="outline"
          onClick={() => {
            s.reset();
            setTheme("dark");
            toast.success("Settings reset to defaults");
          }}
        >
          <RefreshCw className="mr-2 h-4 w-4" />
          Reset to defaults
        </Button>
      </div>

      <div className="rounded-xl border border-border/40 bg-card/30 p-4 text-center text-xs text-muted-foreground">
        Need to start over?{" "}
        <button
          onClick={() => setView("landing")}
          className="font-medium text-indigo-400 hover:underline"
        >
          Back to landing
        </button>
      </div>
    </div>
  );
}

function SectionHeader({
  icon: Icon,
  title,
  description,
}: {
  icon: typeof Sun;
  title: string;
  description: string;
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500/20 to-purple-500/20 ring-1 ring-inset ring-indigo-500/30">
        <Icon className="h-4 w-4 text-indigo-400" />
      </div>
      <div>
        <h2 className="font-display text-lg font-semibold">{title}</h2>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
    </div>
  );
}

function ToggleRow({
  icon: Icon,
  label,
  description,
  checked,
  onCheckedChange,
}: {
  icon: typeof Sun;
  label: string;
  description: string;
  checked: boolean;
  onCheckedChange: (v: boolean) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-lg px-2 py-3 transition-colors hover:bg-muted/30">
      <div className="flex items-start gap-3">
        <Icon className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
        <div>
          <Label className="text-sm font-medium">{label}</Label>
          <p className="text-xs text-muted-foreground">{description}</p>
        </div>
      </div>
      <Switch checked={checked} onCheckedChange={onCheckedChange} />
    </div>
  );
}
