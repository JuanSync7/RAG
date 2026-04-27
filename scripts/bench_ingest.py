#!/usr/bin/env python3
"""Minimal ingest benchmark: CPU vs GPU wall-clock on a fixed document set.

Intentionally disables optional stages (KG, vision, LLM metadata, docling,
document store) so the measured work is dominated by chunking, embedding,
and vector-store insert — the paths that the device choice actually affects.

Run as:
    LD_LIBRARY_PATH=<nvidia_libs> CUDA_VISIBLE_DEVICES=0 \\
        uv run --no-sync python scripts/bench_ingest.py --device gpu --limit 30

Writes a JSON result to scripts/.bench_results/<device>-<timestamp>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


def _bootstrap_env(host_mode: bool) -> None:
    # Point at the containerized rag-weaviate (must be `compose up rag-weaviate`).
    os.environ.setdefault("RAG_WEAVIATE_MODE", "networked")
    os.environ.setdefault("RAG_WEAVIATE_HOST", "localhost")
    os.environ.setdefault("RAG_WEAVIATE_HTTP_PORT", "8090")
    os.environ.setdefault("RAG_WEAVIATE_GRPC_PORT", "50051")
    # Keep observability quiet; we time things ourselves.
    os.environ.setdefault("RAG_OBSERVABILITY_PROVIDER", "noop")
    # Force local embedding backend (not vLLM).
    os.environ.setdefault("RAG_INFERENCE_BACKEND", "local")
    # Avoid unrelated heavyweights.
    os.environ.setdefault("RAG_INGESTION_VERBOSE_STAGE_LOGS", "false")
    os.environ.setdefault("KG_ENABLED", "false")


def _pick_files(docs_dir: Path, limit: int, seed: int = 42) -> list[Path]:
    candidates = sorted(
        p for p in docs_dir.rglob("*.md")
        if p.is_file() and p.stat().st_size > 0
    )
    if limit and limit < len(candidates):
        rng = random.Random(seed)
        candidates = rng.sample(candidates, limit)
        candidates.sort()
    return candidates


def _load_params(path: Path) -> dict:
    """Load tunable params if the file exists; else return {}."""
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def run_benchmark(docs_dir: Path, files: list[Path], device_label: str, params: dict) -> dict:
    # Imports after env is set so config.settings picks up overrides.
    import torch  # noqa: PLC0415

    from src.ingest.common.types import IngestionConfig  # noqa: PLC0415
    from src.ingest.impl import ingest_directory  # noqa: PLC0415

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"

    # Mutable knobs auto-research can tune. Defaults match current production values.
    tunable = {
        "chunk_size": max(64, min(int(params.get("chunk_size", 512)), 4096)),
        "chunk_overlap": max(0, min(int(params.get("chunk_overlap", 50)), 512)),
        "hybrid_chunker_max_tokens": max(128, min(int(params.get("hybrid_chunker_max_tokens", 512)), 4096)),
        "semantic_chunking": bool(params.get("semantic_chunking", False)),
        "min_chunk_chars": max(1, min(int(params.get("min_chunk_chars", 40)), 2048)),
        "min_quality_score": max(0.0, min(float(params.get("min_quality_score", 0.45)), 1.0)),
        "embedding_batch_size": max(8, min(int(params.get("embedding_batch_size", 64)), 512)),
    }

    config = IngestionConfig(
        enable_llm_metadata=False,
        enable_docling_parser=False,
        enable_vision_processing=False,
        enable_multimodal_processing=False,
        enable_document_refactoring=False,
        enable_cross_reference_extraction=False,
        enable_knowledge_graph_extraction=False,
        enable_knowledge_graph_storage=False,
        enable_quality_validation=False,
        build_kg=False,
        export_processed=False,
        persist_refactor_mirror=False,
        store_documents=False,
        enable_visual_embedding=False,
        **tunable,
    )

    total_bytes = sum(p.stat().st_size for p in files)
    t0 = time.monotonic()
    summary = ingest_directory(
        documents_dir=docs_dir,
        config=config,
        fresh=True,
        update=False,
        selected_sources=files,
        batch_id=f"bench-{device_label}-{int(t0)}",
    )
    elapsed = time.monotonic() - t0

    return {
        "device_label": device_label,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "total_kb": round(total_bytes / 1024, 1),
        "elapsed_sec": round(elapsed, 3),
        "docs_per_sec": round(len(files) / elapsed, 3) if elapsed > 0 else None,
        "kb_per_sec": round((total_bytes / 1024) / elapsed, 3) if elapsed > 0 else None,
        "processed": getattr(summary, "processed", None),
        "skipped": getattr(summary, "skipped", None),
        "failed": getattr(summary, "failed", None),
        "stored_chunks": getattr(summary, "stored_chunks", None),
        "chunks_per_sec": (
            round(getattr(summary, "stored_chunks", 0) / elapsed, 3)
            if elapsed > 0 and getattr(summary, "stored_chunks", 0)
            else None
        ),
        "params": tunable,
        "errors": list(getattr(summary, "errors", []) or [])[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("scripts/.bench_results"))
    parser.add_argument(
        "--params-file",
        type=Path,
        default=Path("scripts/bench_params.json"),
        help="JSON of tunable IngestionConfig knobs (auto-research's mutable surface).",
    )
    args = parser.parse_args()

    if args.device == "cpu":
        # Monkey-patch torch.cuda.is_available() to False BEFORE any ingest code
        # imports torch. CUDA_VISIBLE_DEVICES={"",-1} triggers a double-free in the
        # WSL2 NVIDIA driver with torch 2.8 — patching is the only reliable path.
        import torch  # noqa: PLC0415
        torch.cuda.is_available = lambda: False
        torch.cuda.device_count = lambda: 0

    _bootstrap_env(host_mode=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    docs_dir = args.docs_dir.resolve()
    if not docs_dir.exists():
        print(f"docs dir missing: {docs_dir}", file=sys.stderr)
        return 2

    files = _pick_files(docs_dir, args.limit, seed=args.seed)
    if not files:
        print(f"no .md files found under {docs_dir}", file=sys.stderr)
        return 2

    params = _load_params(args.params_file)
    print(f"[bench] device={args.device} files={len(files)} docs_dir={docs_dir} params={params}")
    result = run_benchmark(docs_dir, files, args.device, params)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.device}-{int(time.time())}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[bench] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
