import type { Metadata } from "next";
import { Inter, Geist_Mono, Sora } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as SonnerToaster } from "@/components/ui/sonner";
import { ThemeProvider } from "@/components/theme-provider";

const inter = Inter({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

const sora = Sora({
  variable: "--font-display",
  subsets: ["latin"],
  display: "swap",
  weight: ["400", "500", "600", "700", "800"],
});

export const metadata: Metadata = {
  title: "CaptionAI — AI Image Caption Generator",
  description:
    "Generate natural language descriptions for any image using Deep Learning. Powered by DenseNet201 encoder and LSTM decoder trained on Flickr8K.",
  keywords: [
    "Image Captioning",
    "Deep Learning",
    "DenseNet201",
    "LSTM",
    "TensorFlow",
    "Flickr8K",
    "AI",
    "Computer Vision",
  ],
  authors: [{ name: "CaptionAI" }],
  openGraph: {
    title: "CaptionAI — AI Image Caption Generator",
    description:
      "Generate natural language descriptions for any image using Deep Learning.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "CaptionAI",
    description:
      "Generate natural language descriptions for any image using Deep Learning.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta name="color-scheme" content="dark light" />
      </head>
      <body
        className={`${inter.variable} ${geistMono.variable} ${sora.variable} antialiased bg-background text-foreground`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="dark"
          enableSystem
          disableTransitionOnChange={false}
        >
          {children}
          <Toaster />
          <SonnerToaster
            position="bottom-right"
            theme="dark"
            richColors
            closeButton
          />
        </ThemeProvider>
      </body>
    </html>
  );
}
