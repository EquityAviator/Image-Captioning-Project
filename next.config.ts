import type { NextConfig } from "next";

const BACKEND = "http://localhost:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  typescript: {
    ignoreBuildErrors: true,
  },
  reactStrictMode: false,
  async rewrites() {
    return [
      { source: "/health", destination: `${BACKEND}/health` },
      { source: "/model-info", destination: `${BACKEND}/model-info` },
      { source: "/predict", destination: `${BACKEND}/predict` },
      { source: "/predict-batch", destination: `${BACKEND}/predict-batch` },
      { source: "/providers/:path*", destination: `${BACKEND}/providers/:path*` },
    ];
  },
};

export default nextConfig;
