"use client";

import { motion, AnimatePresence } from "framer-motion";
import { Trash2, RotateCw, Search, History as HistoryIcon, ImageIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { useHistory } from "@/lib/history-store";
import { useState, useMemo } from "react";
import { formatDistanceToNow } from "date-fns";
import { toast } from "sonner";

interface Props {
  onReuse?: (imageDataUrl: string) => void;
  compact?: boolean;
}

export function HistoryPanel({ onReuse, compact }: Props) {
  const entries = useHistory((s) => s.entries);
  const remove = useHistory((s) => s.remove);
  const clear = useHistory((s) => s.clear);
  const [q, setQ] = useState("");

  const filtered = useMemo(() => {
    const ql = q.toLowerCase().trim();
    if (!ql) return entries;
    return entries.filter(
      (e) =>
        e.caption.toLowerCase().includes(ql) ||
        (e.filename || "").toLowerCase().includes(ql),
    );
  }, [entries, q]);

  if (compact && entries.length === 0) {
    return (
      <Card className="border-border/40 bg-card/40 p-6 backdrop-blur-sm">
        <div className="flex flex-col items-center justify-center gap-2 text-center">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-muted/40">
            <HistoryIcon className="h-5 w-5 text-muted-foreground" />
          </div>
          <p className="text-sm font-medium">No history yet</p>
          <p className="text-xs text-muted-foreground">
            Your previous predictions will appear here.
          </p>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-3">
      {!compact && (
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search history…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              className="h-9 border-border/40 bg-card/40 pl-9 backdrop-blur-sm"
            />
          </div>
          {entries.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                clear();
                toast.success("History cleared");
              }}
              className="h-9"
            >
              <Trash2 className="mr-1.5 h-3.5 w-3.5" /> Clear
            </Button>
          )}
        </div>
      )}

      <div className={compact ? "max-h-80 space-y-3 overflow-y-auto pr-1" : "grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3"}>
        <AnimatePresence>
          {filtered.length === 0 ? (
            <Card className="col-span-full border-border/40 bg-card/40 p-8 text-center backdrop-blur-sm">
              <ImageIcon className="mx-auto h-8 w-8 text-muted-foreground" />
              <p className="mt-2 text-sm text-muted-foreground">
                {q ? "No matching results" : "No predictions yet"}
              </p>
            </Card>
          ) : (
            filtered.map((e) => (
              <motion.div
                key={e.id}
                layout
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.25 }}
              >
                <Card className="group overflow-hidden border-border/40 bg-card/40 backdrop-blur-sm transition-colors hover:border-border/80 hover:bg-card/60">
                  <div className="relative aspect-video w-full overflow-hidden bg-muted/20">
                    {/* history thumbnail */}
                    <img
                      src={e.image}
                      alt={e.caption}
                      className="h-full w-full object-cover transition-transform group-hover:scale-105"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-background/80 via-background/0 to-background/0" />
                    <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between gap-2 text-[10px] text-muted-foreground">
                      <span className="flex items-center gap-1 rounded bg-background/60 px-1.5 py-0.5 backdrop-blur-sm">
                        <ImageIcon className="h-2.5 w-2.5" />
                        {e.inference_time}
                      </span>
                      <span className="rounded bg-background/60 px-1.5 py-0.5 backdrop-blur-sm">
                        {formatDistanceToNow(e.created_at, { addSuffix: true })}
                      </span>
                    </div>
                  </div>
                  <div className="p-3">
                    <p className="line-clamp-2 text-sm leading-snug">{e.caption}</p>
                    <div className="mt-3 flex items-center gap-1.5">
                      {onReuse && (
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => onReuse(e.image)}
                          className="h-7 px-2 text-xs"
                        >
                          <RotateCw className="mr-1 h-3 w-3" /> Reuse
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => {
                          remove(e.id);
                          toast.success("Entry deleted");
                        }}
                        className="ml-auto h-7 px-2 text-xs text-muted-foreground hover:text-rose-400"
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                </Card>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
