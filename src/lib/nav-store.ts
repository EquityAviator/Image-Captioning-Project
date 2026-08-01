/**
 * Navigation store — single-page app view switching.
 */

"use client";

import { create } from "zustand";
import type { ViewKey } from "./types";

interface NavStore {
  view: ViewKey;
  setView: (v: ViewKey) => void;
  sidebarOpen: boolean;
  setSidebarOpen: (b: boolean) => void;
}

export const useNav = create<NavStore>()((set) => ({
  view: "landing",
  setView: (v) => set({ view: v }),
  sidebarOpen: false,
  setSidebarOpen: (b) => set({ sidebarOpen: b }),
}));
