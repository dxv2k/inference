import { useEffect, useRef, useState } from "react";
import { useStore } from "../store";
import * as api from "../api";

export function Toolbar({ onToggleRunPanel }: { onToggleRunPanel: () => void }) {
  const workflowName = useStore((s) => s.workflowName);
  const setWorkflowName = useStore((s) => s.setWorkflowName);
  const clear = useStore((s) => s.clear);
  const loadSpec = useStore((s) => s.loadSpec);
  const getCurrentSpec = useStore((s) => s.getCurrentSpec);
  const setValidateStatus = useStore((s) => s.setValidateStatus);
  const validateStatus = useStore((s) => s.validateStatus);
  const setRuntimeParams = useStore((s) => s.setRuntimeParams);

  const [builtin, setBuiltin] = useState<{ id: string; name: string; description?: string }[]>([]);
  const [saved, setSaved] = useState<{ name: string }[]>([]);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.getBuiltinList().then((r) => setBuiltin(r.workflows)).catch(() => setBuiltin([]));
    api.getSaved().then((r) => setSaved(r.workflows)).catch(() => setSaved([]));
  }, []);

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener("click", onClick);
    return () => document.removeEventListener("click", onClick);
  }, []);

  const onLoadBuiltin = async (id: string) => {
    setMenuOpen(false);
    setBusy(true);
    try {
      const r = await api.getBuiltin(id);
      loadSpec(r.workflow, r.name);
      if (r.default_runtime_params) setRuntimeParams(r.default_runtime_params);
    } catch (e: any) {
      alert(`load failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const onLoadSaved = async (name: string) => {
    setMenuOpen(false);
    setBusy(true);
    try {
      const r = await api.loadSaved(name);
      loadSpec(r.workflow, r.name);
    } catch (e: any) {
      alert(`load failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const onSave = async () => {
    const name = prompt("Save as (alphanumerics, hyphens, underscores):", workflowName || "my-workflow");
    if (!name) return;
    const { spec, errors } = getCurrentSpec();
    if (errors.length) {
      if (!confirm(`Serialization warnings:\n\n${errors.join("\n")}\n\nSave anyway?`)) return;
    }
    setBusy(true);
    try {
      await api.save(name, spec);
      setWorkflowName(name);
      const r = await api.getSaved();
      setSaved(r.workflows);
      alert(`saved: saved_workflows/${name}.json`);
    } catch (e: any) {
      alert(`save failed: ${e.message}`);
    } finally {
      setBusy(false);
    }
  };

  const onValidate = async () => {
    const { spec, errors } = getCurrentSpec();
    if (errors.length) {
      setValidateStatus({ ok: false, error: `local errors: ${errors.join("; ")}` });
      return;
    }
    setBusy(true);
    try {
      const r = await api.validate(spec);
      if (r.ok) setValidateStatus({ ok: true });
      else setValidateStatus({ ok: false, error: r.error });
    } catch (e: any) {
      setValidateStatus({ ok: false, error: e.message });
    } finally {
      setBusy(false);
    }
  };

  const onClear = () => {
    if (confirm("Clear the canvas?")) clear();
  };

  return (
    <header className="wf-toolbar">
      <div className="wf-toolbar-left">
        <span className="wf-app-title">Workflow Editor</span>
        <input
          className="wf-toolbar-name"
          type="text"
          placeholder="(unnamed)"
          value={workflowName}
          onChange={(e) => setWorkflowName(e.target.value)}
        />
      </div>
      <div className="wf-toolbar-right">
        <div className="wf-dropdown" ref={menuRef}>
          <button className="wf-btn" disabled={busy} onClick={() => setMenuOpen((v) => !v)}>
            Load ▾
          </button>
          {menuOpen && (
            <div className="wf-dropdown-menu">
              <div className="wf-dropdown-header">Builtin</div>
              {builtin.length === 0 && <div className="wf-dropdown-empty">none</div>}
              {builtin.map((b) => (
                <button key={b.id} className="wf-dropdown-item" onClick={() => onLoadBuiltin(b.id)}>
                  {b.name}
                </button>
              ))}
              <div className="wf-dropdown-header">Saved</div>
              {saved.length === 0 && <div className="wf-dropdown-empty">none</div>}
              {saved.map((s) => (
                <button key={s.name} className="wf-dropdown-item" onClick={() => onLoadSaved(s.name)}>
                  {s.name}
                </button>
              ))}
            </div>
          )}
        </div>
        <button className="wf-btn" disabled={busy} onClick={onSave}>Save</button>
        <button className="wf-btn" disabled={busy} onClick={onValidate}>Validate</button>
        <button className="wf-btn wf-btn-primary" disabled={busy} onClick={onToggleRunPanel}>Run</button>
        <button className="wf-btn wf-btn-danger" disabled={busy} onClick={onClear}>Clear</button>
        {validateStatus && (
          <span className={`wf-status ${validateStatus.ok ? "wf-status-ok" : "wf-status-err"}`}>
            {validateStatus.ok ? "ok" : validateStatus.error}
          </span>
        )}
      </div>
    </header>
  );
}
