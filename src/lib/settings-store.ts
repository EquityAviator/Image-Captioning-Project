/**
 * Settings store (persisted to localStorage).
 */

"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { DEFAULT_SETTINGS, type AppSettings, type ThemeMode, type InferenceProvider } from "./types";

interface SettingsStore extends AppSettings {
  set: <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => void;
  reset: () => void;
  setTheme: (t: ThemeMode) => void;
  setInferenceProvider: (p: InferenceProvider) => void;
}

export const useSettings = create<SettingsStore>()(
  persist(
    (set) => ({
      ...DEFAULT_SETTINGS,
      set: (key, value) => set({ [key]: value } as Partial<AppSettings>),
      reset: () => set({ ...DEFAULT_SETTINGS }),
      setTheme: (t) => set({ theme: t }),
      setInferenceProvider: (p) => set({ inferenceProvider: p }),
    }),
    {
      name: "captionai:settings",
      version: 2,
    },
  ),
);
