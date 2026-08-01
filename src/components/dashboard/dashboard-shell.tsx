"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Sidebar } from "@/components/dashboard/sidebar";
import { DashboardNavbar } from "@/components/dashboard/dashboard-navbar";
import { DashboardView } from "@/components/dashboard/dashboard-view";
import { ModelInfoView } from "@/components/dashboard/model-info-view";
import { SettingsView } from "@/components/dashboard/settings-view";
import { HistoryPanel } from "@/components/dashboard/history-panel";
import { Navbar } from "@/components/navbar";
import { Footer } from "@/components/footer";
import { useNav } from "@/lib/nav-store";
import { useSettings } from "@/lib/settings-store";

export function DashboardShell() {
  const view = useNav((s) => s.view);
  const setView = useNav((s) => s.setView);
  const animations = useSettings((s) => s.animationsEnabled);

  return (
    <div className="flex min-h-screen flex-col">
      <Navbar />
      <div className="flex flex-1">
        <Sidebar />
        <div className="flex flex-1 flex-col overflow-hidden">
          <DashboardNavbar />
          <main className="flex-1 overflow-y-auto px-4 pb-12 pt-6 sm:px-6 lg:px-8 lg:pt-8">
            <div className="mx-auto max-w-7xl">
              <AnimatePresence mode="wait">
                <motion.div
                  key={view}
                  initial={animations ? { opacity: 0, y: 16 } : false}
                  animate={{ opacity: 1, y: 0 }}
                  exit={animations ? { opacity: 0, y: -8 } : undefined}
                  transition={{ duration: 0.25, ease: "easeOut" }}
                >
                  {view === "dashboard" && <DashboardView />}
                  {view === "model-info" && <ModelInfoView />}
                  {view === "settings" && <SettingsView />}
                  {view === "history" && (
                    <div className="space-y-6">
                      <div>
                        <h1 className="font-display text-3xl font-bold tracking-tight">
                          History
                        </h1>
                        <p className="mt-1 text-sm text-muted-foreground">
                          All your previous predictions, stored locally.
                        </p>
                      </div>
                      <HistoryPanel />
                    </div>
                  )}
                </motion.div>
              </AnimatePresence>
            </div>
          </main>
        </div>
      </div>
      <Footer />
    </div>
  );
}
