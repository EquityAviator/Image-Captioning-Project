/**
 * Settings store (persisted to localStorage).
 */

"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { DEFAULT_SETTINGS, type AppSettings, type ThemeMode } from "./types";

interface SettingsStore extends AppSettings {
  set: <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => void;
  reset: () => void;
  setTheme: (t: ThemeMode) => void;
}

export const useSettings = create<SettingsStore>()(
  persist(
    (set) => ({
      ...DEFAULT_SETTINGS,
      set: (key, value) => set({ [key]: value } as Partial<AppSettings>),
      reset: () => set({ ...DEFAULT_SETTINGS }),
      setTheme: (t) => set({ theme: t }),
    }),
    {
      name: "captionai:settings",
      version: 1,
    },
  ),
);
