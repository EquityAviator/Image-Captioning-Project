"use client";

import {
  LayoutDashboard,
  Sparkles,
  History,
  Info,
  Settings,
  PanelLeftClose,
  PanelLeft,
  Github,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { useNav } from "@/lib/nav-store";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import type { ViewKey } from "@/lib/types";

const items: { key: ViewKey; label: string; icon: typeof LayoutDashboard }[] = [
  { key: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { key: "history", label: "History", icon: History },
  { key: "model-info", label: "Model Info", icon: Info },
  { key: "settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const view = useNav((s) => s.view);
  const setView = useNav((s) => s.setView);
  const sidebarOpen = useNav((s) => s.sidebarOpen);
  const setSidebarOpen = useNav((s) => s.setSidebarOpen);

  return (
    <>
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-background/60 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-border/40 bg-card/40 backdrop-blur-xl transition-transform duration-300 lg:sticky lg:top-0 lg:z-0 lg:h-screen lg:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        {/* Logo */}
        <div className="flex h-16 items-center justify-between border-b border-border/40 px-4">
          <button
            onClick={() => setView("landing")}
            className="group flex items-center gap-2.5"
          >
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 via-purple-500 to-blue-500 shadow-lg shadow-indigo-500/30 transition-transform group-hover:scale-110">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div className="flex flex-col items-start">
              <span className="font-display text-sm font-bold leading-none">
                CaptionAI
              </span>
              <span className="text-[10px] uppercase tracking-widest text-muted-foreground">
                Dashboard
              </span>
            </div>
          </button>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden"
            aria-label="Close sidebar"
          >
            <PanelLeftClose className="h-4 w-4" />
          </Button>
        </div>

        {/* Generate Caption button */}
        <div className="p-3">
          <Button
            onClick={() => setView("dashboard")}
            className="w-full justify-start bg-gradient-to-r from-indigo-500 to-purple-500 text-white hover:from-indigo-600 hover:to-purple-600"
          >
            <Sparkles className="mr-2 h-4 w-4" />
            Generate Caption
          </Button>
        </div>

        {/* Nav items */}
        <nav className="flex-1 space-y-1 px-3 py-2">
          {items.map((it) => {
            const active = view === it.key;
            return (
              <button
                key={it.key}
                onClick={() => {
                  setView(it.key);
                  setSidebarOpen(false);
                }}
                className={cn(
                  "group relative flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  active
                    ? "bg-indigo-500/10 text-indigo-400"
                    : "text-muted-foreground hover:bg-muted/60 hover:text-foreground",
                )}
              >
                {active && (
                  <motion.div
                    layoutId="sidebar-active"
                    className="absolute inset-0 rounded-lg bg-indigo-500/10 ring-1 ring-inset ring-indigo-500/30"
                  />
                )}
                <it.icon className="relative h-4 w-4" />
                <span className="relative">{it.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="border-t border-border/40 p-3">
          <Button
            variant="ghost"
            size="sm"
            asChild
            className="w-full justify-start text-muted-foreground"
          >
            <a href="https://github.com" target="_blank" rel="noopener noreferrer">
              <Github className="mr-2 h-4 w-4" />
              View Source
            </a>
          </Button>
        </div>
      </aside>

      {/* Mobile trigger */}
      <Button
        variant="ghost"
        size="icon"
        onClick={() => setSidebarOpen(true)}
        className="fixed left-4 top-20 z-30 lg:hidden"
        aria-label="Open sidebar"
      >
        <PanelLeft className="h-5 w-5" />
      </Button>
    </>
  );
}
