import { useCallback, useMemo, useRef } from "react";
import ReactFlow, {
  Background,
  Controls,
  type Connection,
  type ReactFlowInstance,
} from "reactflow";
import "reactflow/dist/style.css";
import { useStore } from "../store";
import { BlockNode } from "./BlockNode";
import { InputNode } from "./InputNode";
import { OutputNode } from "./OutputNode";

export function Canvas() {
  const nodes = useStore((s) => s.nodes);
  const edges = useStore((s) => s.edges);
  const applyNodeChanges = useStore((s) => s.applyNodeChanges);
  const applyEdgeChanges = useStore((s) => s.applyEdgeChanges);
  const connect = useStore((s) => s.connect);
  const setSelected = useStore((s) => s.setSelected);
  const addBlock = useStore((s) => s.addBlock);

  const wrapper = useRef<HTMLDivElement>(null);
  const rfInstance = useRef<ReactFlowInstance | null>(null);

  const nodeTypes = useMemo(
    () => ({ block: BlockNode, input: InputNode, output: OutputNode }),
    [],
  );

  const onConnect = useCallback(
    (c: Connection) => {
      if (!c.source || !c.target || !c.sourceHandle || !c.targetHandle) return;
      connect({
        source: c.source,
        sourceHandle: c.sourceHandle,
        target: c.target,
        targetHandle: c.targetHandle,
      });
    },
    [connect],
  );

  const onDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
  }, []);

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      const blockType = e.dataTransfer.getData("application/x-block-type");
      if (!blockType) return;
      const bounds = wrapper.current?.getBoundingClientRect();
      if (!bounds || !rfInstance.current) return;
      const position = rfInstance.current.screenToFlowPosition({
        x: e.clientX,
        y: e.clientY,
      });
      addBlock(blockType, position);
    },
    [addBlock],
  );

  const onSelectionChange = useCallback(
    ({ nodes }: { nodes: any[] }) => {
      if (nodes && nodes.length > 0) setSelected(nodes[0].id);
      else setSelected(null);
    },
    [setSelected],
  );

  return (
    <div className="wf-canvas" ref={wrapper} onDragOver={onDragOver} onDrop={onDrop}>
      <ReactFlow
        nodes={nodes as any}
        edges={edges as any}
        nodeTypes={nodeTypes}
        onNodesChange={applyNodeChanges}
        onEdgesChange={applyEdgeChanges}
        onConnect={onConnect}
        onInit={(inst) => (rfInstance.current = inst)}
        onSelectionChange={onSelectionChange as any}
        fitView
        deleteKeyCode={["Backspace", "Delete"]}
      >
        <Background gap={16} size={1} />
        <Controls />
      </ReactFlow>
    </div>
  );
}
