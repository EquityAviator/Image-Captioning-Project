"use client";

import { motion, AnimatePresence } from "framer-motion";
import { LandingPage } from "@/components/landing/landing-page";
import { DashboardShell } from "@/components/dashboard/dashboard-shell";
import { Navbar } from "@/components/navbar";
import { useNav } from "@/lib/nav-store";
import { useSettings } from "@/lib/settings-store";

export default function Home() {
  const view = useNav((s) => s.view);
  const animations = useSettings((s) => s.animationsEnabled);
  const isLanding = view === "landing";

  return (
    <AnimatePresence mode="wait">
      <motion.div
        key={isLanding ? "landing" : "app"}
        initial={animations ? { opacity: 0 } : false}
        animate={{ opacity: 1 }}
        exit={animations ? { opacity: 0 } : undefined}
        transition={{ duration: 0.3, ease: "easeInOut" }}
      >
        {isLanding ? (
          <div>
            <Navbar />
            <LandingPage />
          </div>
        ) : (
          <DashboardShell />
        )}
      </motion.div>
    </AnimatePresence>
  );
}
