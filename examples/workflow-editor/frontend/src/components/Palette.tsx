import { useMemo, useState } from "react";
import { useStore } from "../store";
import type { ManifestBlock } from "../types";

export function Palette() {
  const manifest = useStore((s) => s.manifest);
  const [filter, setFilter] = useState("");
  const addInput = useStore((s) => s.addInput);
  const addOutput = useStore((s) => s.addOutput);
  const nodes = useStore((s) => s.nodes);

  const grouped = useMemo(() => {
    const m = manifest?.blocks ?? [];
    const f = filter.toLowerCase();
    const filtered = m.filter(
      (b) =>
        !f ||
        b.type.toLowerCase().includes(f) ||
        b.display_name.toLowerCase().includes(f) ||
        b.short_description.toLowerCase().includes(f),
    );
    const by: Record<string, ManifestBlock[]> = {};
    for (const b of filtered) {
      if (!by[b.category]) by[b.category] = [];
      by[b.category].push(b);
    }
    for (const k of Object.keys(by)) by[k].sort((a, b) => a.display_name.localeCompare(b.display_name));
    return by;
  }, [manifest, filter]);

  const onDragStart = (e: React.DragEvent<HTMLDivElement>, blockType: string) => {
    e.dataTransfer.setData("application/x-block-type", blockType);
    e.dataTransfer.effectAllowed = "move";
  };

  const nextName = (prefix: string) => {
    const taken = new Set<string>();
    for (const n of nodes) {
      if (n.type === "input") taken.add(n.data.name);
      else if (n.type === "output") taken.add(n.data.name);
    }
    let i = 1;
    let n = prefix;
    while (taken.has(n)) {
      i++;
      n = `${prefix}_${i}`;
    }
    return n;
  };

  return (
    <aside className="wf-palette">
      <h3 className="wf-palette-title">Blocks</h3>
      <input
        className="wf-palette-search"
        type="text"
        placeholder="filter..."
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
      />
      <div className="wf-palette-special">
        <div className="wf-palette-cat">graph i/o</div>
        <button className="wf-palette-tile wf-palette-input" onClick={() => addInput("WorkflowImage", nextName("image"))}>
          + Input: Image
        </button>
        <button className="wf-palette-tile wf-palette-input" onClick={() => addInput("WorkflowParameter", nextName("param"))}>
          + Input: Param
        </button>
        <button className="wf-palette-tile wf-palette-output" onClick={() => addOutput(nextName("output"))}>
          + Output
        </button>
      </div>
      {Object.keys(grouped)
        .sort()
        .map((cat) => (
          <div key={cat} className="wf-palette-group">
            <div className="wf-palette-cat">{cat}</div>
            {grouped[cat].map((b) => (
              <div
                key={b.type}
                className="wf-palette-tile"
                draggable
                title={b.short_description}
                onDragStart={(e) => onDragStart(e, b.type)}
              >
                {b.display_name}
              </div>
            ))}
          </div>
        ))}
    </aside>
  );
}
