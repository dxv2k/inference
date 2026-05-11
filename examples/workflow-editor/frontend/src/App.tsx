import { useEffect, useState } from "react";
import { ReactFlowProvider } from "reactflow";
import { Toolbar } from "./components/Toolbar";
import { Palette } from "./components/Palette";
import { Canvas } from "./components/Canvas";
import { Inspector } from "./components/Inspector";
import { RunPanel } from "./components/RunPanel";
import { useStore } from "./store";
import * as api from "./api";

export default function App() {
  const setManifest = useStore((s) => s.setManifest);
  const manifest = useStore((s) => s.manifest);
  const [runOpen, setRunOpen] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    api
      .getBlocks()
      .then((m) => setManifest(m))
      .catch((e) => setLoadError(`failed to load blocks manifest: ${e.message}`));
  }, [setManifest]);

  return (
    <ReactFlowProvider>
      <div className="wf-app">
        <Toolbar onToggleRunPanel={() => setRunOpen((v) => !v)} />
        {loadError && <div className="wf-load-error">{loadError}</div>}
        <main className="wf-main">
          <Palette />
          <Canvas />
          <Inspector />
        </main>
        <RunPanel open={runOpen} onClose={() => setRunOpen(false)} />
        {!manifest && !loadError && (
          <div className="wf-loading">loading block manifest...</div>
        )}
      </div>
    </ReactFlowProvider>
  );
}
