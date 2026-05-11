// All fetch() helpers to the backend. Base URL is empty -- same origin in prod;
// the Vite dev server proxies /api -> :7873.

import type {
  Manifest,
  BuiltinEntry,
  SavedEntry,
  ValidateResponse,
  RunResponse,
  WorkflowSpec,
} from "./types";

async function jsonOrThrow<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let detail = "";
    try {
      detail = JSON.stringify(await r.json());
    } catch {
      detail = await r.text();
    }
    throw new Error(`HTTP ${r.status}: ${detail || r.statusText}`);
  }
  return r.json() as Promise<T>;
}

export async function getHealth(): Promise<{ status: string; plugins_loaded: string[] }> {
  return jsonOrThrow(await fetch("/api/health"));
}

export async function getBlocks(): Promise<Manifest> {
  return jsonOrThrow(await fetch("/api/blocks"));
}

export async function getBuiltinList(): Promise<{ workflows: BuiltinEntry[] }> {
  return jsonOrThrow(await fetch("/api/workflows/builtin"));
}

export async function getBuiltin(
  id: string,
): Promise<{ id: string; name: string; workflow: WorkflowSpec; default_runtime_params?: Record<string, unknown> }> {
  return jsonOrThrow(await fetch(`/api/workflows/builtin/${encodeURIComponent(id)}`));
}

export async function getSaved(): Promise<{ workflows: SavedEntry[] }> {
  return jsonOrThrow(await fetch("/api/workflows/saved"));
}

export async function loadSaved(
  name: string,
): Promise<{ name: string; workflow: WorkflowSpec }> {
  return jsonOrThrow(await fetch(`/api/workflows/saved/${encodeURIComponent(name)}`));
}

export async function save(
  name: string,
  workflow: WorkflowSpec,
): Promise<{ name: string; path: string }> {
  return jsonOrThrow(
    await fetch("/api/workflows/save", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, workflow }),
    }),
  );
}

export async function validate(workflow: WorkflowSpec): Promise<ValidateResponse> {
  return jsonOrThrow(
    await fetch("/api/workflows/validate", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ workflow }),
    }),
  );
}

export async function run(
  workflow: WorkflowSpec,
  runtimeParams: Record<string, unknown>,
  imageFile: File | null,
  imagePath: string | null,
): Promise<RunResponse> {
  const fd = new FormData();
  fd.append("workflow", JSON.stringify(workflow));
  fd.append("runtime_params", JSON.stringify(runtimeParams));
  if (imageFile) fd.append("image", imageFile);
  else if (imagePath) fd.append("image_path", imagePath);
  return jsonOrThrow(await fetch("/api/workflows/run", { method: "POST", body: fd }));
}
