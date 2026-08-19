/**
 * useCaption — wraps the predict API with progress stages + abort support.
 */

"use client";

import { useCallback, useRef, useState } from "react";
import { apiClient, ApiError } from "@/lib/api";
import { useSettings } from "@/lib/settings-store";
import { useHistory } from "@/lib/history-store";
import type { PredictResponse } from "@/lib/types";

export type StageKey =
  | "idle"
  | "uploading"
  | "extracting"
  | "encoding"
  | "decoding"
  | "done"
  | "error";

export interface Stage {
  key: StageKey;
  label: string;
  description: string;
}

export const STAGES: Stage[] = [
  { key: "uploading", label: "Uploading", description: "Sending image to the API…" },
  { key: "extracting", label: "Extracting Features", description: "Running DenseNet201 encoder…" },
  { key: "encoding", label: "Encoding", description: "Projecting image features to 256-dim…" },
  { key: "decoding", label: "Generating Caption", description: "Running LSTM decoder word by word…" },
  { key: "done", label: "Done", description: "Caption ready!" },
];

interface UseCaptionReturn {
  isLoading: boolean;
  stage: StageKey;
  stageIndex: number; // 0..4, -1 if idle
  result: PredictResponse | null;
  error: string | null;
  predict: (file: File | Blob, filename?: string) => Promise<PredictResponse | null>;
  reset: () => void;
  abort: () => void;
}

export function useCaption(): UseCaptionReturn {
  const [isLoading, setIsLoading] = useState(false);
  const [stage, setStage] = useState<StageKey>("idle");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const apiBaseUrl = useSettings((s) => s.apiBaseUrl);
  const inferenceProvider = useSettings((s) => s.inferenceProvider);
  const historyEnabled = useSettings((s) => s.historyEnabled);
  const addHistory = useHistory((s) => s.add);

  const stageIndex = STAGES.findIndex((s) => s.key === stage);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsLoading(false);
    setStage("idle");
    setResult(null);
    setError(null);
  }, []);

  const abort = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsLoading(false);
    setStage("idle");
  }, []);

  const predict = useCallback(
    async (file: File | Blob, filename = "image.jpg") => {
      // reset state
      setError(null);
      setResult(null);
      setIsLoading(true);
      setStage("uploading");

      // Simulated stage timers — give the UI a chance to show each stage
      // even when the backend returns quickly. They never block the actual
      // API call (which runs in parallel).
      const stageTimers: number[] = [];
      const scheduleStage = (key: StageKey, delay: number) => {
        const t = window.setTimeout(() => {
          setStage((cur) => (cur === "error" || cur === "done" ? cur : key));
        }, delay);
        stageTimers.push(t);
      };
      scheduleStage("extracting", 400);
      scheduleStage("encoding", 1100);
      scheduleStage("decoding", 1900);

      const ac = new AbortController();
      abortRef.current = ac;

      try {
        // Read file as dataURL for history (only if history is enabled)
        let dataUrl: string | null = null;
        if (historyEnabled) {
          dataUrl = await blobToDataUrl(file);
        }

        const res = await apiClient.predict(file, filename, apiBaseUrl || undefined, ac.signal, inferenceProvider);
        // Clear pending stage timers
        stageTimers.forEach((t) => clearTimeout(t));
        setStage("done");
        setResult(res);
        setIsLoading(false);

        if (historyEnabled && dataUrl && res.success) {
          addHistory({
            id: crypto.randomUUID(),
            image: dataUrl,
            caption: res.caption,
            inference_time: res.inference_time,
            confidence: res.confidence,
            provider: res.provider,
            created_at: Date.now(),
            filename,
          });
        }
        return res;
      } catch (err) {
        stageTimers.forEach((t) => clearTimeout(t));
        if (err instanceof ApiError) {
          setError(err.detail || err.message);
        } else if (err instanceof Error && err.name === "AbortError") {
          setError("Cancelled");
        } else if (err instanceof Error) {
          setError(err.message);
        } else {
          setError("Unknown error");
        }
        setStage("error");
        setIsLoading(false);
        return null;
      }
    },
    [apiBaseUrl, inferenceProvider, historyEnabled, addHistory],
  );

  return { isLoading, stage, stageIndex, result, error, predict, reset, abort };
}

function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result as string);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });
}
