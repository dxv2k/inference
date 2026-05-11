import { memo } from "react";
import { Handle, Position } from "reactflow";
import type { NodeProps } from "reactflow";
import type { OutputNodeData } from "../types";

function OutputNodeImpl({ data, selected }: NodeProps<OutputNodeData>) {
  return (
    <div className={`wf-output ${selected ? "wf-block-selected" : ""}`}>
      <Handle
        type="target"
        position={Position.Left}
        id="in:value"
        className="wf-handle wf-handle-in"
      />
      <div className="wf-output-kind">Output</div>
      <div className="wf-output-name">{data.name}</div>
    </div>
  );
}

export const OutputNode = memo(OutputNodeImpl);
