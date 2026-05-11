import { memo } from "react";
import { Handle, Position } from "reactflow";
import type { NodeProps } from "reactflow";
import type { InputNodeData } from "../types";

function InputNodeImpl({ data, selected }: NodeProps<InputNodeData>) {
  return (
    <div className={`wf-input ${selected ? "wf-block-selected" : ""}`}>
      <div className="wf-input-kind">{data.inputKind === "WorkflowImage" ? "Image" : "Param"}</div>
      <div className="wf-input-name">{data.name}</div>
      <Handle
        type="source"
        position={Position.Right}
        id="out:value"
        className="wf-handle wf-handle-out"
      />
    </div>
  );
}

export const InputNode = memo(InputNodeImpl);
