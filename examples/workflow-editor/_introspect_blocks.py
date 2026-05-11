"""One-shot introspection script: produce blocks_manifest.json for the editor.

Run with:
    cd /media/ubuntu_data/viAct/inference/examples/workflow-editor
    PYTHONPATH=../local-workflow-demo:.. python _introspect_blocks.py

The script reuses real_engine_runner.py's plugin-registration trick (sets
WORKFLOWS_PLUGINS before importing inference) and then asks the introspection
API for every block. We filter down to a curated allowlist for the editor.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Mirror real_engine_runner.py: register custom plugins before importing inference.
HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "local-workflow-demo"
for p in (str(DEMO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _triton_available() -> bool:
    try:
        import tritonclient.grpc  # noqa: F401
        return True
    except ImportError:
        return False


def _sam3_available() -> bool:
    try:
        import sam3  # noqa: F401
        return True
    except ImportError:
        return False


_PLUGINS = ["local_yolo_plugin", "yolo_world_plugin"]
if _sam3_available():
    _PLUGINS.append("sam3_plugin")
if _triton_available():
    _PLUGINS.append("optimize.triton.triton_yolo_plugin")
os.environ.setdefault("WORKFLOWS_PLUGINS", ",".join(_PLUGINS))

from inference.core.workflows.execution_engine.introspection.blocks_loader import (  # noqa: E402
    describe_available_blocks,
)

# Curated allowlist. Keep the palette focused — the user said "15-25 blocks".
ALLOWLIST = {
    # custom plugins (always present)
    "local_models/ultralytics_yolo@v1",
    "local_models/yolo_world@v1",
    "local_models/sam3@v1",
    "triton/yolo@v1",
    # Roboflow core_steps subset
    "roboflow_core/trackers_bytetrack@v1",
    "roboflow_core/velocity@v1",
    "roboflow_core/time_in_zone@v2",
    "roboflow_core/byte_tracker@v1",  # legacy alias just in case
    "roboflow_core/bounding_box_visualization@v1",
    "roboflow_core/label_visualization@v1",
    "roboflow_core/polygon_zone_visualization@v1",
    "roboflow_core/mask_visualization@v1",
    "roboflow_core/line_counter@v2",
    "roboflow_core/line_counter_visualization@v1",
    "roboflow_core/webhook_sink@v1",
    "roboflow_core/openai_compatible@v1",
    "roboflow_core/expression@v1",
    "roboflow_core/property_definition@v1",
    "roboflow_core/detections_filter@v1",
    "roboflow_core/detections_classes_replacement@v1",
    "roboflow_core/dynamic_crop@v1",
    "roboflow_core/continue_if@v1",
    "roboflow_core/first_non_empty_or_default@v1",
    "roboflow_core/csv_formatter@v1",
}


def _category_for(block_schema: dict, identifier: str) -> str:
    json_extra = (block_schema.get("name") or "").lower()
    block_type = (block_schema.get("block_type") or "").lower()
    if block_type:
        return block_type
    if "visualization" in identifier:
        return "visualization"
    if "tracker" in identifier or "velocity" in identifier or "time_in_zone" in identifier:
        return "transformation"
    if "sink" in identifier or "webhook" in identifier or "csv" in identifier:
        return "sink"
    if "openai" in identifier or "vlm" in identifier:
        return "model"
    if "filter" in identifier or "crop" in identifier or "expression" in identifier:
        return "transformation"
    return "logic"


def _extract_inputs(block_schema: dict, manifest_metadata) -> list[dict]:
    """Convert a block manifest's Pydantic JSON schema into editor-friendly input descriptors.

    For each property in the manifest (except `type` and `name`):
      - read the JSON-schema entry
      - if the property name is a SelectorDefinition (from manifest_metadata),
        record allowed reference kinds
      - otherwise record `primitive` type from the JSON schema
    """
    props: dict = block_schema.get("properties", {}) or {}
    required = set(block_schema.get("required", []) or [])
    selectors_map = getattr(manifest_metadata, "selectors", {}) or {}
    primitives_map = getattr(manifest_metadata, "primitive_types", {}) or {}

    inputs = []
    for name, schema in props.items():
        if name in {"type", "name"}:
            continue
        item: dict = {
            "name": name,
            "required": name in required,
            "description": schema.get("description") or "",
            "default": schema.get("default"),
        }

        # Selector-capable input? Record allowed reference kinds.
        sel = selectors_map.get(name)
        if sel is not None:
            kinds: set[str] = set()
            for ref in sel.allowed_references:
                kinds.add(ref.selected_element)
                for k in ref.kind:
                    kinds.add(k.name)
            item["accepts_selector"] = True
            item["selector_kinds"] = sorted(kinds)
            item["is_list_element"] = sel.is_list_element
            item["is_dict_element"] = sel.is_dict_element
        else:
            item["accepts_selector"] = False
            item["selector_kinds"] = []

        # Primitive type hint (Pydantic's JSON-schema "type" or anyOf).
        primitive_kinds: list[str] = []
        if "type" in schema:
            primitive_kinds.append(schema["type"])
        if "anyOf" in schema:
            for variant in schema["anyOf"]:
                if "type" in variant:
                    primitive_kinds.append(variant["type"])
                if "$ref" in variant:
                    primitive_kinds.append(variant["$ref"].split("/")[-1])
        prim = primitives_map.get(name)
        if prim is not None:
            item["primitive_annotation"] = prim.type_annotation
        item["primitive_kinds"] = sorted(set(primitive_kinds))

        if "enum" in schema:
            item["enum"] = schema["enum"]

        inputs.append(item)
    return inputs


def _extract_outputs(outputs_manifest) -> list[dict]:
    out = []
    for od in outputs_manifest:
        out.append({
            "name": od.name,
            "kinds": [k.name for k in (od.kind or [])],
        })
    return out


def main():
    from inference.core.workflows.execution_engine.introspection.entities import (
        BlockManifestMetadata,
    )
    from inference.core.workflows.execution_engine.introspection.schema_parser import (
        parse_block_manifest,
    )

    def get_manifest_metadata(manifest_cls):
        return parse_block_manifest(manifest_type=manifest_cls)

    description = describe_available_blocks(dynamic_blocks=[])
    all_blocks = description.blocks

    by_type = {b.manifest_type_identifier: b for b in all_blocks}
    for b in all_blocks:
        for alias in b.manifest_type_identifier_aliases:
            by_type.setdefault(alias, b)

    # Resolve allowlist into actual BlockDescription objects.
    selected = []
    missing = []
    for ident in ALLOWLIST:
        if ident in by_type:
            selected.append((ident, by_type[ident]))
        else:
            missing.append(ident)

    print(f"[introspect] {len(all_blocks)} total blocks available")
    print(f"[introspect] {len(selected)} matched from allowlist")
    if missing:
        print(f"[introspect] {len(missing)} allowlist entries NOT FOUND (skipping):")
        for m in missing:
            print(f"             - {m}")

    out_blocks = []
    for ident, b in selected:
        try:
            md = get_manifest_metadata(b.manifest_class)
        except Exception as e:
            print(f"[introspect] could not introspect manifest for {ident}: {e}")
            md = BlockManifestMetadata(primitive_types={}, selectors={})

        out_blocks.append({
            "type": ident,
            "aliases": b.manifest_type_identifier_aliases,
            "display_name": b.human_friendly_block_name,
            "category": _category_for(b.block_schema, ident),
            "block_source": b.block_source,
            "short_description": (b.block_schema.get("short_description")
                                  or b.block_schema.get("description") or ""),
            "long_description": b.block_schema.get("long_description") or "",
            "inputs": _extract_inputs(b.block_schema, md),
            "outputs": _extract_outputs(b.outputs_manifest),
            "execution_engine_compatibility": b.execution_engine_compatibility,
            "accepts_batch_input": b.manifest_class.accepts_batch_input(),
            "parameters_accepting_batches":
                list(b.manifest_class.get_parameters_accepting_batches()),
        })

    # Stable sort: category first, then display_name.
    out_blocks.sort(key=lambda x: (x["category"], x["display_name"]))

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_available": len(all_blocks),
        "blocks": out_blocks,
        "categories": sorted({b["category"] for b in out_blocks}),
    }

    out_path = HERE / "blocks_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[introspect] wrote {out_path} with {len(out_blocks)} blocks")


if __name__ == "__main__":
    main()
