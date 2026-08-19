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

function safeSetItem(key: string, value: string) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // Quota exceeded — silently ignore. The in-memory state is still updated
    // so the UI stays consistent; only persistence is skipped.
  }
}

export const useHistory = create<HistoryStore>()(
  persist(
    (set, get) => ({
      entries: [],
      add: (entry) => {
        const next = [entry, ...get().entries].slice(0, MAX_ENTRIES);
        set({ entries: next });
      },
      remove: (id) => {
        set({ entries: get().entries.filter((e) => e.id !== id) });
      },
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
      storage: {
        getItem: (name) => {
          try {
            const raw = localStorage.getItem(name);
            return raw ? JSON.parse(raw) : null;
          } catch {
            // If stored data is corrupt or too large to parse, clear it
            try { localStorage.removeItem(name); } catch {}
            return null;
          }
        },
        setItem: (name, value) => safeSetItem(name, JSON.stringify(value)),
        removeItem: (name) => {
          try { localStorage.removeItem(name); } catch {}
        },
      },
    },
  ),
);
