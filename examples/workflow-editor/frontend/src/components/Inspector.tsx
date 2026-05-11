import { useMemo, useState } from "react";
import { useStore } from "../store";
import type { ManifestInput, BlockNodeData, InputNodeData, OutputNodeData } from "../types";

function JsonField({
  value,
  onChange,
  rows = 3,
}: {
  value: unknown;
  onChange: (v: unknown) => void;
  rows?: number;
}) {
  const initial = useMemo(() => {
    if (value === undefined) return "";
    if (typeof value === "string") return JSON.stringify(value);
    return JSON.stringify(value, null, 2);
  }, [value]);
  const [text, setText] = useState(initial);
  const [err, setErr] = useState<string | null>(null);
  // Re-init when external value changes
  const valueKey = JSON.stringify(value);
  const [seenKey, setSeenKey] = useState(valueKey);
  if (seenKey !== valueKey) {
    setSeenKey(valueKey);
    setText(initial);
    setErr(null);
  }
  return (
    <>
      <textarea
        className="wf-input-textarea"
        rows={rows}
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          try {
            const v = e.target.value.trim() === "" ? undefined : JSON.parse(e.target.value);
            onChange(v);
            setErr(null);
          } catch (ex: any) {
            setErr(ex.message);
          }
        }}
      />
      {err && <div className="wf-input-err">{err}</div>}
    </>
  );
}

function StringField({
  value,
  onChange,
  multiline = false,
}: {
  value: unknown;
  onChange: (v: unknown) => void;
  multiline?: boolean;
}) {
  const str = value == null ? "" : String(value);
  if (multiline) {
    return (
      <textarea
        className="wf-input-textarea"
        rows={4}
        value={str}
        onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value)}
      />
    );
  }
  return (
    <input
      className="wf-input-text"
      type="text"
      value={str}
      onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value)}
    />
  );
}

function NumberField({ value, onChange }: { value: unknown; onChange: (v: unknown) => void }) {
  const str = value == null ? "" : String(value);
  return (
    <input
      className="wf-input-text"
      type="number"
      value={str}
      step="any"
      onChange={(e) => {
        const v = e.target.value;
        if (v === "") onChange(undefined);
        else {
          const n = Number(v);
          onChange(Number.isFinite(n) ? n : v);
        }
      }}
    />
  );
}

function BoolField({ value, onChange }: { value: unknown; onChange: (v: unknown) => void }) {
  return (
    <input
      type="checkbox"
      checked={Boolean(value)}
      onChange={(e) => onChange(e.target.checked)}
    />
  );
}

function ParamEditor({
  field,
  value,
  onChange,
}: {
  field: ManifestInput;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  // Dict / list fields -> JSON textarea
  if (field.is_dict_element || field.is_list_element) {
    return <JsonField value={value} onChange={onChange} rows={4} />;
  }
  const kinds = field.primitive_kinds || [];
  // Enum
  if (field.enum && field.enum.length > 0) {
    return (
      <select
        className="wf-input-select"
        value={value == null ? "" : String(value)}
        onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value)}
      >
        <option value="">(default)</option>
        {field.enum.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    );
  }
  if (kinds.includes("integer") || kinds.includes("number") || kinds.includes("float")) {
    return <NumberField value={value} onChange={onChange} />;
  }
  if (kinds.includes("boolean")) {
    return <BoolField value={value} onChange={onChange} />;
  }
  if (kinds.includes("array") || kinds.includes("object")) {
    return <JsonField value={value} onChange={onChange} rows={4} />;
  }
  // Default to string. Use multiline for long strings.
  const isLong =
    typeof value === "string" && (value.length > 60 || value.includes("\n"));
  return <StringField value={value} onChange={onChange} multiline={isLong} />;
}

export function Inspector() {
  const selectedId = useStore((s) => s.selectedNodeId);
  const nodes = useStore((s) => s.nodes);
  const edges = useStore((s) => s.edges);
  const manifest = useStore((s) => s.manifest);
  const setParam = useStore((s) => s.setParam);
  const setStepName = useStore((s) => s.setStepName);
  const setInputName = useStore((s) => s.setInputName);
  const setInputDefault = useStore((s) => s.setInputDefault);
  const setOutputName = useStore((s) => s.setOutputName);
  const deleteNode = useStore((s) => s.deleteNode);

  if (!selectedId) {
    return (
      <aside className="wf-inspector">
        <div className="wf-inspector-empty">Select a node to edit its parameters.</div>
      </aside>
    );
  }
  const node = nodes.find((n) => n.id === selectedId);
  if (!node) {
    return <aside className="wf-inspector"><div className="wf-inspector-empty">no node</div></aside>;
  }

  if (node.type === "input") {
    const d = node.data as InputNodeData;
    return (
      <aside className="wf-inspector">
        <div className="wf-inspector-header">
          <span>{d.inputKind === "WorkflowImage" ? "Image Input" : "Parameter Input"}</span>
          <button className="wf-btn-mini wf-btn-danger" onClick={() => deleteNode(selectedId)}>delete</button>
        </div>
        <div className="wf-field">
          <label>name</label>
          <input
            className="wf-input-text"
            type="text"
            value={d.name}
            onChange={(e) => setInputName(selectedId, e.target.value)}
          />
        </div>
        {d.inputKind === "WorkflowParameter" && (
          <div className="wf-field">
            <label>default_value (JSON)</label>
            <JsonField
              value={d.default_value}
              onChange={(v) => setInputDefault(selectedId, v)}
              rows={3}
            />
          </div>
        )}
      </aside>
    );
  }

  if (node.type === "output") {
    const d = node.data as OutputNodeData;
    return (
      <aside className="wf-inspector">
        <div className="wf-inspector-header">
          <span>Output</span>
          <button className="wf-btn-mini wf-btn-danger" onClick={() => deleteNode(selectedId)}>delete</button>
        </div>
        <div className="wf-field">
          <label>name</label>
          <input
            className="wf-input-text"
            type="text"
            value={d.name}
            onChange={(e) => setOutputName(selectedId, e.target.value)}
          />
        </div>
      </aside>
    );
  }

  // Block node
  const d = node.data as BlockNodeData;
  const block = manifest?.blocks.find((b) => b.type === d.blockType);
  // Compute wired field names (planner #1: hide fields that have an incoming edge)
  const wired = new Set<string>();
  for (const e of edges) {
    if (e.target !== selectedId) continue;
    if (!e.targetHandle.startsWith("in:")) continue;
    wired.add(e.targetHandle.slice(3));
  }

  return (
    <aside className="wf-inspector">
      <div className="wf-inspector-header">
        <span title={d.blockType}>{block?.display_name ?? d.blockType}</span>
        <button className="wf-btn-mini wf-btn-danger" onClick={() => deleteNode(selectedId)}>delete</button>
      </div>
      <div className="wf-inspector-type">{d.blockType}</div>
      <div className="wf-field">
        <label>name (step id)</label>
        <input
          className="wf-input-text"
          type="text"
          value={d.stepName}
          onChange={(e) => setStepName(selectedId, e.target.value)}
        />
      </div>
      {block ? (
        <>
          <h4 className="wf-section-title">Inputs</h4>
          {block.inputs.length === 0 && <div className="wf-empty-msg">no inputs declared</div>}
          {block.inputs.map((field) => {
            const isWired = wired.has(field.name);
            return (
              <div className={`wf-field ${isWired ? "wf-field-wired" : ""}`} key={field.name}>
                <label title={field.description}>
                  {field.name}
                  {field.required ? " *" : ""}
                </label>
                {isWired ? (
                  <div className="wf-field-wired-hint">connected (edit by disconnecting the wire)</div>
                ) : (
                  <ParamEditor
                    field={field}
                    value={d.params[field.name]}
                    onChange={(v) => setParam(selectedId, field.name, v)}
                  />
                )}
              </div>
            );
          })}
        </>
      ) : (
        <div className="wf-empty-msg">unknown block (not in manifest)</div>
      )}
    </aside>
  );
}
