"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Check, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { STAGES, type StageKey } from "@/hooks/use-caption";

interface Props {
  currentStage: StageKey;
  stageIndex: number;
}

export function PredictionStages({ currentStage, stageIndex }: Props) {
  const isError = currentStage === "error";
  const isDone = currentStage === "done";

  return (
    <div className="rounded-2xl border border-border/40 bg-card/40 p-5 backdrop-blur-sm">
      <div className="mb-4 flex items-center justify-between">
        <p className="font-display text-sm font-semibold">
          {isError ? "Prediction failed" : isDone ? "Prediction complete" : "Processing"}
        </p>
        <span className="text-xs text-muted-foreground">
          {isDone ? "100%" : `${Math.min(stageIndex + 1, STAGES.length)}/${STAGES.length}`}
        </span>
      </div>

      <div className="space-y-3">
        {STAGES.slice(0, 4).map((s, i) => {
          const isActive = !isError && !isDone && i === stageIndex;
          const isCompleted = isDone || (!isError && i < stageIndex);
          const isPending = !isActive && !isCompleted;
          return (
            <div
              key={s.key}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 transition-colors",
                isActive && "bg-indigo-500/10",
                isCompleted && "opacity-90",
                isPending && "opacity-40",
              )}
            >
              <div
                className={cn(
                  "flex h-7 w-7 shrink-0 items-center justify-center rounded-full border transition-colors",
                  isCompleted && "border-emerald-500/30 bg-emerald-500/15 text-emerald-400",
                  isActive && "border-indigo-500/30 bg-indigo-500/15 text-indigo-400",
                  isPending && "border-border bg-muted/30 text-muted-foreground",
                )}
              >
                <AnimatePresence mode="wait">
                  {isCompleted ? (
                    <motion.div
                      key="check"
                      initial={{ scale: 0, rotate: -45 }}
                      animate={{ scale: 1, rotate: 0 }}
                      exit={{ scale: 0 }}
                    >
                      <Check className="h-3.5 w-3.5" />
                    </motion.div>
                  ) : isActive ? (
                    <motion.div
                      key="spinner"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    </motion.div>
                  ) : (
                    <motion.span
                      key="num"
                      className="text-xs font-semibold"
                    >
                      {i + 1}
                    </motion.span>
                  )}
                </AnimatePresence>
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium leading-none">{s.label}</p>
                <p className="mt-1 text-xs text-muted-foreground">{s.description}</p>
              </div>
              {isActive && (
                <motion.div
                  className="h-1 w-12 overflow-hidden rounded-full bg-muted"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                >
                  <motion.div
                    className="h-full bg-gradient-to-r from-indigo-500 to-purple-500"
                    initial={{ x: "-100%" }}
                    animate={{ x: "100%" }}
                    transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
                  />
                </motion.div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
