"use client";

import { Hero } from "@/components/landing/hero";
import { Features } from "@/components/landing/features";
import { Architecture } from "@/components/landing/architecture";
import { HowItWorks } from "@/components/landing/how-it-works";
import { ModelSection } from "@/components/landing/model-section";
import { Performance } from "@/components/landing/performance";
import { DemoCTA } from "@/components/landing/demo-cta";
import { Footer } from "@/components/footer";

export function LandingPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <main className="flex-1">
        <Hero />
        <Features />
        <Architecture />
        <HowItWorks />
        <ModelSection />
        <Performance />
        <DemoCTA />
      </main>
      <Footer />
    </div>
  );
}
