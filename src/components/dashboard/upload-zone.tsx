"use client";

import { useCallback, useRef, useState, type DragEvent } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  UploadCloud,
  Image as ImageIcon,
  X,
  RefreshCw,
  Sparkles,
  FileImage,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface UploadZoneProps {
  onFileSelected: (file: File) => void;
  onReset: () => void;
  previewUrl: string | null;
  isLoading: boolean;
  disabled?: boolean;
}

const ACCEPTED = ["image/png", "image/jpeg", "image/jpg", "image/webp"];
const MAX_BYTES = 10 * 1024 * 1024;

export function UploadZone({
  onFileSelected,
  onReset,
  previewUrl,
  isLoading,
  disabled,
}: UploadZoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validate = useCallback((file: File): string | null => {
    if (!ACCEPTED.includes(file.type)) {
      return `Unsupported file type: ${file.type}. Use PNG, JPG, or JPEG.`;
    }
    if (file.size > MAX_BYTES) {
      return `File too large: ${(file.size / 1024 / 1024).toFixed(1)} MB. Max 10 MB.`;
    }
    return null;
  }, []);

  const handleFile = useCallback(
    (file: File) => {
      const err = validate(file);
      if (err) {
        setError(err);
        return;
      }
      setError(null);
      onFileSelected(file);
    },
    [onFileSelected, validate],
  );

  const onDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled || isLoading) return;
      const file = e.dataTransfer.files?.[0];
      if (file) handleFile(file);
    },
    [disabled, isLoading, handleFile],
  );

  const onInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
      // reset input value so selecting the same file again re-triggers
      e.target.value = "";
    },
    [handleFile],
  );

  return (
    <div className="w-full">
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(",")}
        onChange={onInputChange}
        className="hidden"
        aria-label="Upload image file"
      />

      <AnimatePresence mode="wait">
        {previewUrl ? (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ duration: 0.3 }}
            className="relative overflow-hidden rounded-2xl border border-border/40 bg-card/40 backdrop-blur-sm"
          >
            {/* Image */}
            <div className="relative aspect-video w-full overflow-hidden bg-muted/20">
              {/* preview image */}
              <img
                src={previewUrl}
                alt="Uploaded preview"
                className="h-full w-full object-contain"
              />

              {/* Loading overlay */}
              <AnimatePresence>
                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-background/70 backdrop-blur-md"
                  >
                    <div className="relative">
                      <div className="h-12 w-12 animate-spin rounded-full border-2 border-indigo-500/30 border-t-indigo-500" />
                      <Sparkles className="absolute inset-0 m-auto h-5 w-5 text-indigo-400" />
                    </div>
                    <p className="text-sm font-medium text-foreground">
                      Processing image…
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Reset button */}
              {!isLoading && (
                <Button
                  size="icon"
                  variant="secondary"
                  onClick={onReset}
                  className="absolute right-3 top-3 h-8 w-8 rounded-full bg-background/80 shadow-md backdrop-blur-md"
                  aria-label="Remove image"
                >
                  <X className="h-4 w-4" />
                </Button>
              )}
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between gap-3 border-t border-border/40 p-3">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <FileImage className="h-3.5 w-3.5" />
                <span>Image ready for captioning</span>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={onReset}
                  disabled={isLoading}
                  className="h-8"
                >
                  <RefreshCw className="mr-1.5 h-3 w-3" />
                  Reset
                </Button>
                <Button
                  size="sm"
                  onClick={() => inputRef.current?.click()}
                  disabled={isLoading}
                  className="h-8 bg-gradient-to-r from-indigo-500 to-purple-500 text-white hover:from-indigo-600 hover:to-purple-600"
                >
                  <UploadCloud className="mr-1.5 h-3 w-3" />
                  Change
                </Button>
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ duration: 0.3 }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            onClick={() => !disabled && inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                inputRef.current?.click();
              }
            }}
            aria-label="Upload an image by clicking or dragging"
            className={cn(
              "group relative flex aspect-video w-full cursor-pointer flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed p-8 text-center transition-all",
              dragOver
                ? "border-indigo-500 bg-indigo-500/10"
                : "border-border/60 bg-card/30 hover:border-indigo-500/50 hover:bg-card/50",
              disabled && "pointer-events-none opacity-50",
            )}
          >
            {/* Animated bg blob */}
            <div
              className={cn(
                "pointer-events-none absolute inset-0 rounded-2xl bg-gradient-to-br from-indigo-500/0 via-purple-500/0 to-blue-500/0 transition-opacity",
                dragOver && "from-indigo-500/10 via-purple-500/10 to-blue-500/10",
              )}
            />

            <motion.div
              animate={
                dragOver
                  ? { scale: 1.1, y: -4 }
                  : { scale: 1, y: 0 }
              }
              className="relative flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 ring-1 ring-inset ring-indigo-500/30"
            >
              <UploadCloud className="h-9 w-9 text-indigo-400" />
            </motion.div>

            <div className="relative">
              <p className="font-display text-lg font-semibold">
                {dragOver ? "Drop your image here" : "Drag & drop an image"}
              </p>
              <p className="mt-1 text-sm text-muted-foreground">
                or{" "}
                <span className="font-medium text-indigo-400 underline-offset-4 group-hover:underline">
                  browse your files
                </span>
              </p>
            </div>

            <div className="relative flex flex-wrap items-center justify-center gap-2 text-xs text-muted-foreground">
              {["PNG", "JPG", "JPEG", "WEBP"].map((t) => (
                <span
                  key={t}
                  className="rounded-md border border-border/40 bg-muted/30 px-2 py-0.5"
                >
                  {t}
                </span>
              ))}
              <span className="text-muted-foreground/60">•</span>
              <span>Max 10 MB</span>
            </div>

            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 8 }}
                  className="relative mt-2 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-1.5 text-xs text-rose-400"
                >
                  {error}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
