import type { NextConfig } from "next";

const BACKEND = "http://localhost:8010";

const nextConfig: NextConfig = {
  // standalone tracing walks the whole tree on every dev compile — very
  // expensive inside OneDrive. Only enable it for real production builds.
  ...(process.env.NODE_ENV === "production" ? { output: "standalone" } : {}),
  // Bound module resolution to the common ancestor of this project AND its
  // node_modules junction target (%LOCALAPPDATA%\capai-local). Files outside
  // are not resolved/watched; the junction stays resolvable.
  turbopack: {
    root: "C:/Users/mh562",
  },
  experimental: {
    // Evict Turbopack's in-memory copies after each persistent-cache
    // snapshot. Without this the dev compiler grew past 7.7 GB RSS /
    // 16 GB commit and OOM'd the machine mid-compile.
    turbopackMemoryEviction: "full",
  },
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
