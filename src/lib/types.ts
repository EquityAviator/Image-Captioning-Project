/**
 * CaptionAI — shared TypeScript types.
 */

export interface PredictResponse {
  caption: string;
  inference_time: string;
  confidence: number;
  provider: string;
  success: boolean;
  error?: string | null;
}

export interface BatchPredictItem {
  filename: string;
  caption: string;
  inference_time: string;
  confidence: number;
  success: boolean;
  error?: string | null;
}

export interface BatchPredictResponse {
  results: BatchPredictItem[];
  total: number;
  success_count: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  provider: string;
  weights_loaded: boolean;
  backend: string;
}

export interface ModelInfoResponse {
  ready: boolean;
  active_provider: string;
  weights_loaded: boolean;
  provider: string;
  encoder: string;
  encoder_feature_dim: number;
  decoder: string;
  embed_dim: number;
  lstm_units: number;
  dense_units: number;
  dropout: number;
  training_dataset: string;
  image_size: number;
  vocab_size: number;
  max_length: number;
  tensorflow_version: string;
  training_metadata: Record<string, unknown>;
}

export interface HistoryEntry {
  id: string;
  image: string; // dataURL
  caption: string;
  inference_time: string;
  confidence: number;
  provider: string;
  created_at: number;
  filename?: string;
}

export type ThemeMode = "dark" | "light" | "system";

export type InferenceProvider = "auto" | "notebook-tensorflow" | "huggingface-blip";

export interface AppSettings {
  theme: ThemeMode;
  animationsEnabled: boolean;
  historyEnabled: boolean;
  typingEffectEnabled: boolean;
  speechSynthesisEnabled: boolean;
  apiBaseUrl: string;
  autoCopyEnabled: boolean;
  inferenceProvider: InferenceProvider;
}

export const DEFAULT_SETTINGS: AppSettings = {
  theme: "dark",
  animationsEnabled: true,
  historyEnabled: true,
  typingEffectEnabled: true,
  speechSynthesisEnabled: false,
  apiBaseUrl: "",
  autoCopyEnabled: false,
  inferenceProvider: "auto",
};

export type ViewKey =
  | "landing"
  | "dashboard"
  | "history"
  | "model-info"
  | "settings";

export interface PredictionStage {
  key: string;
  label: string;
  description: string;
}
