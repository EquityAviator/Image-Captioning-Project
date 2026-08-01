/**
 * History store (persisted to localStorage, capped to last 50 entries).
 */

"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { HistoryEntry } from "./types";

interface HistoryStore {
  entries: HistoryEntry[];
  add: (entry: HistoryEntry) => void;
  remove: (id: string) => void;
  clear: () => void;
  getById: (id: string) => HistoryEntry | undefined;
  search: (q: string) => HistoryEntry[];
}

const MAX_ENTRIES = 50;

export const useHistory = create<HistoryStore>()(
  persist(
    (set, get) => ({
      entries: [],
      add: (entry) =>
        set((s) => ({
          entries: [entry, ...s.entries].slice(0, MAX_ENTRIES),
        })),
      remove: (id) =>
        set((s) => ({ entries: s.entries.filter((e) => e.id !== id) })),
      clear: () => set({ entries: [] }),
      getById: (id) => get().entries.find((e) => e.id === id),
      search: (q) => {
        const ql = q.toLowerCase().trim();
        if (!ql) return get().entries;
        return get().entries.filter(
          (e) =>
            e.caption.toLowerCase().includes(ql) ||
            (e.filename || "").toLowerCase().includes(ql),
        );
      },
    }),
    {
      name: "captionai:history",
      version: 1,
      // Don't store huge dataURLs forever — cap to last 50 by slicing above.
    },
  ),
);
