/**
 * CaptionAI API client.
 *
 * All requests go through the Caddy gateway at port 81 with the
 * `XTransformPort=8000` query parameter so the browser never talks to
 * the FastAPI backend directly.
 */

import type {
  BatchPredictResponse,
  HealthResponse,
  ModelInfoResponse,
  PredictResponse,
} from "./types";

const BACKEND_PORT = "8000";

function buildUrl(path: string, baseUrl?: string): string {
  // If user has configured a custom API URL in Settings, use it directly.
  if (baseUrl && baseUrl.trim().length > 0) {
    return `${baseUrl.replace(/\/$/, "")}${path}`;
  }
  // Otherwise route via the Caddy gateway using XTransformPort.
  const hasQuery = path.includes("?");
  const sep = hasQuery ? "&" : "?";
  return `${path}${sep}XTransformPort=${BACKEND_PORT}`;
}

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(`API error ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export const apiClient = {
  async getHealth(baseUrl?: string): Promise<HealthResponse> {
    const res = await fetch(buildUrl("/health", baseUrl), {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!res.ok) throw new ApiError(res.status, await safeErr(res));
    return res.json();
  },

  async getModelInfo(baseUrl?: string): Promise<ModelInfoResponse> {
    const res = await fetch(buildUrl("/model-info", baseUrl), {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!res.ok) throw new ApiError(res.status, await safeErr(res));
    return res.json();
  },

  async predict(
    file: File | Blob,
    filename = "image.jpg",
    baseUrl?: string,
    signal?: AbortSignal,
  ): Promise<PredictResponse> {
    const form = new FormData();
    form.append("file", file, filename);
    const res = await fetch(buildUrl("/predict", baseUrl), {
      method: "POST",
      body: form,
      signal,
    });
    if (!res.ok) throw new ApiError(res.status, await safeErr(res));
    return res.json();
  },

  async predictBatch(
    files: File[],
    baseUrl?: string,
    signal?: AbortSignal,
  ): Promise<BatchPredictResponse> {
    const form = new FormData();
    for (const f of files) form.append("files", f);
    const res = await fetch(buildUrl("/predict-batch", baseUrl), {
      method: "POST",
      body: form,
      signal,
    });
    if (!res.ok) throw new ApiError(res.status, await safeErr(res));
    return res.json();
  },
};

async function safeErr(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return data?.detail || res.statusText;
  } catch {
    return res.statusText;
  }
}
