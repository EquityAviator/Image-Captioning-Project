"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Sparkles, ImagePlus, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { UploadZone } from "@/components/dashboard/upload-zone";
import { PredictionStages } from "@/components/dashboard/prediction-stages";
import { PredictionCard } from "@/components/dashboard/prediction-card";
import { HistoryPanel } from "@/components/dashboard/history-panel";
import { useCaption } from "@/hooks/use-caption";
import { useNav } from "@/lib/nav-store";
import { apiClient } from "@/lib/api";
import type { HealthResponse } from "@/lib/types";

export function DashboardView() {
  const setView = useNav((s) => s.setView);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const { isLoading, stage, stageIndex, result, error, predict, reset } = useCaption();

  useEffect(() => {
    apiClient
      .getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const onFileSelected = (f: File) => {
    setFile(f);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(f));
  };

  const onReset = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setFile(null);
    reset();
  };

  const onPredict = async () => {
    if (!file) return;
    await predict(file, file.name);
  };

  const onRegenerate = async () => {
    if (!file) return;
    // re-run with the same file
    await predict(file, file.name);
  };

  const onReuse = (imageDataUrl: string) => {
    setPreviewUrl(imageDataUrl);
    // Convert dataURL back to File for re-prediction
    fetch(imageDataUrl)
      .then((r) => r.blob())
      .then((blob) => {
        const f = new File([blob], "reuse.jpg", { type: blob.type });
        setFile(f);
      });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between"
      >
        <div>
          <h1 className="font-display text-3xl font-bold tracking-tight">
            Generate Caption
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Upload an image and let the model describe it for you.
          </p>
        </div>
        {health && (
          <div className="flex items-center gap-2 rounded-lg border border-border/40 bg-card/40 px-3 py-1.5 text-xs backdrop-blur-sm">
            <span
              className={`h-1.5 w-1.5 rounded-full ${health.model_loaded ? "bg-emerald-500" : "bg-rose-500"}`}
            />
            <span className="text-muted-foreground">
              Provider: <span className="font-mono text-foreground">{health.provider}</span>
            </span>
          </div>
        )}
      </motion.div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Left column — upload + prediction */}
        <div className="space-y-6 lg:col-span-2">
          <UploadZone
            onFileSelected={onFileSelected}
            onReset={onReset}
            previewUrl={previewUrl}
            isLoading={isLoading}
          />

          {/* Action button row */}
          <AnimatePresence>
            {file && !result && !isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 8 }}
                className="flex items-center justify-between gap-3"
              >
                <p className="text-sm text-muted-foreground">
                  Ready to generate a caption for{" "}
                  <span className="font-medium text-foreground">{file.name}</span>
                </p>
                <Button
                  size="lg"
                  onClick={onPredict}
                  className="bg-gradient-to-r from-indigo-500 via-purple-500 to-blue-500 text-white shadow-lg shadow-indigo-500/30 hover:from-indigo-600 hover:to-purple-600"
                >
                  <Sparkles className="mr-2 h-4 w-4" />
                  Predict Caption
                </Button>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Loading stages */}
          <AnimatePresence>
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 12 }}
              >
                <PredictionStages currentStage={stage} stageIndex={stageIndex} />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Error */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 12 }}
              >
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Prediction failed</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Result */}
          <AnimatePresence>
            {result && result.success && (
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 12 }}
              >
                <PredictionCard
                  result={result}
                  imageUrl={previewUrl}
                  onRegenerate={onRegenerate}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Right column — recent history */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-display text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              Recent Predictions
            </h2>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setView("history")}
              className="h-7 text-xs"
            >
              View all
            </Button>
          </div>
          <HistoryPanel compact onReuse={onReuse} />
        </div>
      </div>
    </div>
  );
}
