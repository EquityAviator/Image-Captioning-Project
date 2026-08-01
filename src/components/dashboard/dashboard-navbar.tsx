"use client";

import { Search, Bell, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { ThemeToggle } from "@/components/theme-toggle";
import { useNav } from "@/lib/nav-store";
import { Badge } from "@/components/ui/badge";

export function DashboardNavbar() {
  const setView = useNav((s) => s.setView);

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center gap-3 border-b border-border/40 bg-background/80 px-4 backdrop-blur-xl sm:px-6">
      {/* Logo (mobile) */}
      <button
        onClick={() => setView("landing")}
        className="flex items-center gap-2 lg:hidden"
      >
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 via-purple-500 to-blue-500">
          <Sparkles className="h-4 w-4 text-white" />
        </div>
      </button>

      {/* Search */}
      <div className="relative flex-1 max-w-md">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          type="search"
          placeholder="Search captions, images…"
          className="h-9 border-border/40 bg-card/40 pl-9 backdrop-blur-sm"
        />
      </div>

      <div className="flex-1" />

      {/* Status pill */}
      <Badge
        variant="outline"
        className="hidden items-center gap-1.5 border-emerald-500/30 bg-emerald-500/10 text-emerald-400 sm:flex"
      >
        <span className="relative flex h-1.5 w-1.5">
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
          <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-emerald-500" />
        </span>
        Model Online
      </Badge>

      <Button variant="ghost" size="icon" className="rounded-full" aria-label="Notifications">
        <Bell className="h-4 w-4" />
      </Button>

      <ThemeToggle />

      <Avatar className="h-9 w-9 ring-2 ring-indigo-500/30">
        <AvatarFallback className="bg-gradient-to-br from-indigo-500 to-purple-500 text-xs font-semibold text-white">
          CA
        </AvatarFallback>
      </Avatar>
    </header>
  );
}
