import { useState } from "react";
import { useStore } from "../store";
import * as api from "../api";

export function RunPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const getCurrentSpec = useStore((s) => s.getCurrentSpec);
  const lastResult = useStore((s) => s.lastResult);
  const setLastResult = useStore((s) => s.setLastResult);
  const runtimeParams = useStore((s) => s.runtimeParams);
  const setRuntimeParams = useStore((s) => s.setRuntimeParams);

  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePath, setImagePath] = useState("");
  const [paramsText, setParamsText] = useState(JSON.stringify(runtimeParams, null, 2));
  const [busy, setBusy] = useState(false);

  if (!open) return null;

  const onRun = async () => {
    const { spec, errors } = getCurrentSpec();
    if (errors.length) {
      if (!confirm(`Serialization warnings:\n\n${errors.join("\n")}\n\nRun anyway?`)) return;
    }
    let parsedParams: Record<string, unknown> = {};
    try {
      parsedParams = paramsText.trim() === "" ? {} : JSON.parse(paramsText);
      setRuntimeParams(parsedParams);
    } catch (e: any) {
      alert(`runtime_params parse error: ${e.message}`);
      return;
    }
    if (!imageFile && !imagePath) {
      alert("Provide an image file OR an image_path (under sample_images/)");
      return;
    }
    setBusy(true);
    try {
      const r = await api.run(spec, parsedParams, imageFile, imagePath || null);
      setLastResult(r);
    } catch (e: any) {
      setLastResult({ ok: false, error: e.message, elapsed_ms: 0 });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="wf-runpanel">
      <div className="wf-runpanel-header">
        <span>Run</span>
        <button className="wf-btn-mini" onClick={onClose}>close ×</button>
      </div>
      <div className="wf-runpanel-body">
        <div className="wf-runpanel-controls">
          <label>Image file:</label>
          <input
            type="file"
            accept="image/jpeg,image/png"
            onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
          />
          <label>or sample path:</label>
          <input
            type="text"
            placeholder="e.g. cars.jpg"
            value={imagePath}
            onChange={(e) => setImagePath(e.target.value)}
          />
          <label>runtime_params (JSON):</label>
          <textarea
            className="wf-runpanel-params"
            rows={5}
            value={paramsText}
            onChange={(e) => setParamsText(e.target.value)}
          />
          <button className="wf-btn wf-btn-primary" disabled={busy} onClick={onRun}>
            {busy ? "running..." : "Run workflow"}
          </button>
        </div>
        <div className="wf-runpanel-output">
          {lastResult ? (
            lastResult.ok ? (
              <>
                <div className="wf-runpanel-meta">elapsed: {lastResult.elapsed_ms} ms</div>
                {lastResult.outputs.map((out, i) => (
                  <div key={i} className="wf-output-card">
                    {out.annotated && (
                      <img
                        className="wf-annotated"
                        alt={`annotated output ${i}`}
                        src={`data:image/png;base64,${out.annotated}`}
                      />
                    )}
                    {out.detections && (
                      <details className="wf-details">
                        <summary>detections</summary>
                        <pre>{JSON.stringify(out.detections, null, 2)}</pre>
                      </details>
                    )}
                    {out.raw_text && (
                      <details className="wf-details" open>
                        <summary>raw_text</summary>
                        <pre>{out.raw_text}</pre>
                      </details>
                    )}
                    <details className="wf-details">
                      <summary>full json</summary>
                      <pre>{JSON.stringify(out, null, 2)}</pre>
                    </details>
                  </div>
                ))}
              </>
            ) : (
              <div className="wf-error">
                <strong>error:</strong> {lastResult.error}
              </div>
            )
          ) : (
            <div className="wf-empty-msg">no result yet</div>
          )}
        </div>
      </div>
    </div>
  );
}
