// Round-trip test: load each builtin spec, deserialize -> serialize -> compare.
//
// Run with: node --experimental-strip-types test/roundtrip.ts
// Requires Node 22+.

import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { workflowJSONToCanvas } from "../src/lib/deserialize.ts";
import { canvasToWorkflowJSON } from "../src/lib/serialize.ts";
import type { Manifest, WorkflowSpec } from "../src/types.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const PARENT = resolve(HERE, "..", "..");

const manifest = JSON.parse(
  readFileSync(resolve(PARENT, "blocks_manifest.json"), "utf-8"),
) as Manifest;

const SPECS_DIR = process.env.WF_SPECS_DIR ?? "/tmp/wf_specs";

const cases = ["speed", "smart", "autoannotate", "sam3_autoannotate", "vlm_only"];

type Diff = { path: string; a: unknown; b: unknown };

function deepEqual(a: unknown, b: unknown, path = "", diffs: Diff[] = []): Diff[] {
  if (a === b) return diffs;
  if (a === null || b === null || typeof a !== typeof b) {
    diffs.push({ path, a, b });
    return diffs;
  }
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) {
      diffs.push({ path: `${path}.length`, a: a.length, b: b.length });
    }
    const n = Math.max(a.length, b.length);
    for (let i = 0; i < n; i++) {
      deepEqual(a[i], b[i], `${path}[${i}]`, diffs);
    }
    return diffs;
  }
  if (typeof a === "object" && typeof b === "object") {
    const ao = a as Record<string, unknown>;
    const bo = b as Record<string, unknown>;
    const ka = Object.keys(ao).sort();
    const kb = Object.keys(bo).sort();
    const allKeys = Array.from(new Set([...ka, ...kb]));
    for (const k of allKeys) {
      if (!(k in ao)) {
        diffs.push({ path: `${path}.${k}`, a: "<missing>", b: bo[k] });
        continue;
      }
      if (!(k in bo)) {
        diffs.push({ path: `${path}.${k}`, a: ao[k], b: "<missing>" });
        continue;
      }
      deepEqual(ao[k], bo[k], `${path}.${k}`, diffs);
    }
    return diffs;
  }
  if (a !== b) diffs.push({ path, a, b });
  return diffs;
}

let allPass = true;
for (const id of cases) {
  const path = resolve(SPECS_DIR, `${id}.json`);
  let original: WorkflowSpec;
  try {
    original = JSON.parse(readFileSync(path, "utf-8")) as WorkflowSpec;
  } catch (e) {
    console.log(`[SKIP] ${id}: cannot read ${path}: ${(e as Error).message}`);
    continue;
  }
  const { nodes, edges, warnings } = workflowJSONToCanvas(original, manifest);
  if (warnings.length) {
    console.log(`  ${id}: deserialize warnings:`);
    for (const w of warnings) console.log(`    - ${w}`);
  }
  const { spec: round, errors } = canvasToWorkflowJSON(nodes, edges, manifest);
  if (errors.length) {
    console.log(`  ${id}: serialize errors:`);
    for (const e of errors) console.log(`    - ${e}`);
  }
  const diffs = deepEqual(original, round);
  if (diffs.length === 0) {
    console.log(`[PASS] ${id}`);
  } else {
    allPass = false;
    console.log(`[FAIL] ${id}: ${diffs.length} differences`);
    for (const d of diffs.slice(0, 12)) {
      console.log(`    ${d.path}: original=${JSON.stringify(d.a)} != round=${JSON.stringify(d.b)}`);
    }
    if (diffs.length > 12) console.log(`    ... and ${diffs.length - 12} more`);
  }
}

if (!allPass) {
  process.exitCode = 1;
}
