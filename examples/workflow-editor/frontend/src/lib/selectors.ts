// Selector parsing helpers — recognise `$inputs.X` and `$steps.X.Y` strings.
//
// We do NOT recurse into dict/list values: a top-level dict like
// `{"image": "$inputs.image"}` is treated as a literal because the planner
// declared that pattern (prompt_parameters) stays a literal in v1.

export type ParsedSelector =
  | { kind: "input"; name: string }
  | { kind: "step"; name: string; field: string };

export function isSelector(value: unknown): value is string {
  return typeof value === "string" && (value.startsWith("$inputs.") || value.startsWith("$steps."));
}

export function parseSelector(s: string): ParsedSelector | null {
  if (s.startsWith("$inputs.")) {
    const name = s.slice("$inputs.".length);
    if (!name) return null;
    return { kind: "input", name };
  }
  if (s.startsWith("$steps.")) {
    const rest = s.slice("$steps.".length);
    const dot = rest.indexOf(".");
    if (dot <= 0) return null;
    const name = rest.slice(0, dot);
    const field = rest.slice(dot + 1);
    if (!name || !field) return null;
    return { kind: "step", name, field };
  }
  return null;
}

export function formatStepSelector(stepName: string, field: string): string {
  return `$steps.${stepName}.${field}`;
}

export function formatInputSelector(name: string): string {
  return `$inputs.${name}`;
}
