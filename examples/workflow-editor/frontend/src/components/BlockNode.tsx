import { memo } from "react";
import { Handle, Position } from "reactflow";
import type { NodeProps } from "reactflow";
import { useStore } from "../store";
import type { BlockNodeData } from "../types";

function BlockNodeImpl({ id, data, selected }: NodeProps<BlockNodeData>) {
  const manifest = useStore((s) => s.manifest);
  const block = manifest?.blocks.find((b) => b.type === data.blockType);
  const inputs = block?.inputs ?? [];
  const outputs = block?.outputs ?? [];
  const displayName = block?.display_name ?? data.blockType;

  return (
    <div className={`wf-block ${selected ? "wf-block-selected" : ""}`}>
      <div className="wf-block-header">
        <span className="wf-block-title">{displayName}</span>
        <span className="wf-block-stepname">{data.stepName}</span>
      </div>
      <div className="wf-block-body">
        <div className="wf-handles-col wf-handles-in">
          {inputs.map((inp) => (
            <div className="wf-handle-row" key={`in-${inp.name}`}>
              <Handle
                type="target"
                position={Position.Left}
                id={`in:${inp.name}`}
                className={`wf-handle wf-handle-in ${inp.required ? "wf-required" : ""}`}
              />
              <span className="wf-handle-label">{inp.name}</span>
            </div>
          ))}
        </div>
        <div className="wf-handles-col wf-handles-out">
          {outputs.map((out) => (
            <div className="wf-handle-row wf-handle-row-out" key={`out-${out.name}`}>
              <span className="wf-handle-label">{out.name}</span>
              <Handle
                type="source"
                position={Position.Right}
                id={`out:${out.name}`}
                className="wf-handle wf-handle-out"
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export const BlockNode = memo(BlockNodeImpl);
