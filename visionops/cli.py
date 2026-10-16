from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

COMMANDS = (
    "ingest",
    "prelabel",
    "select",
    "export-labelstudio",
    "train",
    "evaluate",
    "export-onnx",
    "serve",
    "benchmark",
    "make-showcase",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visionops",
        description="Prepare, evaluate, and serve crowd-person detectors.",
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--input", nargs="+", type=Path, required=True)
    ingest_parser.add_argument("--out", type=Path, required=True)
    ingest_parser.add_argument("--stride", type=int, default=1)
    ingest_parser.add_argument("--max-frames", type=int)
    ingest_parser.add_argument("--seed", type=int, default=0)
    ingest_parser.add_argument("--group")

    prelabel_parser = subparsers.add_parser("prelabel")
    prelabel_parser.add_argument("--manifest", type=Path, required=True)
    prelabel_parser.add_argument("--model", type=Path, required=True)
    prelabel_parser.add_argument("--out", type=Path, required=True)
    prelabel_parser.add_argument("--confidence", type=float, default=0.25)
    prelabel_parser.add_argument("--iou", type=float, default=0.7)
    prelabel_parser.add_argument("--device", default="cpu")

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--predictions", type=Path, required=True)
    select_parser.add_argument("--out", type=Path, required=True)
    select_parser.add_argument("--budget", type=int, required=True)
    select_parser.add_argument("--strategy", choices=("random", "uncertainty"), default="random")
    select_parser.add_argument("--seed", type=int, default=0)

    export_ls_parser = subparsers.add_parser("export-labelstudio")
    export_ls_parser.add_argument("--selection", type=Path, required=True)
    export_ls_parser.add_argument("--out", type=Path, required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--out", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, default=0)
    train_parser.add_argument("--weights", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--device", default="cpu")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", type=Path, required=True)
    evaluate_parser.add_argument("--model", type=Path, required=True)
    evaluate_parser.add_argument("--out", type=Path, required=True)

    export_onnx_parser = subparsers.add_parser("export-onnx")
    export_onnx_parser.add_argument("--model", type=Path, required=True)
    export_onnx_parser.add_argument("--out", type=Path, required=True)
    export_onnx_parser.add_argument("--imgsz", type=int, default=640)
    export_onnx_parser.add_argument("--sample-images", nargs="*", type=Path, default=[])

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--model", type=Path, required=True)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--max-upload-bytes", type=int, default=10_000_000)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--model", type=Path, required=True)
    benchmark_parser.add_argument("--images", nargs="+", type=Path, required=True)
    benchmark_parser.add_argument("--out", type=Path, required=True)
    benchmark_parser.add_argument("--warmup", type=int, default=3)
    benchmark_parser.add_argument("--iterations", type=int, default=20)

    showcase_parser = subparsers.add_parser("make-showcase")
    showcase_parser.add_argument("--video", type=Path, required=True)
    showcase_parser.add_argument("--model", type=Path, required=True)
    showcase_parser.add_argument("--out", type=Path, required=True)
    showcase_parser.add_argument("--start-s", type=float, default=0.0)
    showcase_parser.add_argument("--duration-s", type=float, default=10.0)
    showcase_parser.add_argument("--source-url")
    showcase_parser.add_argument("--license")
    showcase_parser.add_argument("--attribution")
    return parser


def _run_ingest(args: argparse.Namespace) -> int:
    from visionops.ingest import ingest

    manifest = ingest(args.input, args.out, args.stride, args.max_frames, args.seed, args.group)
    print(f"manifest={manifest}")
    return 0


def _run_prelabel(args: argparse.Namespace) -> int:
    from visionops.prelabel import UltralyticsPersonDetector, prelabel_manifest

    if args.model.suffix.lower() == ".onnx":
        from visionops.inference import OnnxPersonDetector

        detector = OnnxPersonDetector(args.model, args.confidence, args.iou)
    else:
        detector = UltralyticsPersonDetector(args.model, args.confidence, args.iou, args.device)
    out = prelabel_manifest(
        args.manifest, detector, args.out, detector.model_id, detector.model_sha256
    )
    print(f"predictions={out}")
    return 0


def _run_select(args: argparse.Namespace) -> int:
    from visionops.jsonl import read_jsonl, write_jsonl
    from visionops.schemas import PredictionRecord
    from visionops.selection import select_records

    predictions = read_jsonl(args.predictions, PredictionRecord)
    records = select_records(
        predictions, args.budget, args.strategy, args.seed, root=args.predictions.parent
    )
    print(f"selection={write_jsonl(args.out, records)}")
    return 0


def _run_export_labelstudio(args: argparse.Namespace) -> int:
    from visionops.annotations import export_labelstudio

    print(f"tasks={export_labelstudio(args.selection, args.out)}")
    return 0


def _run_train(args: argparse.Namespace) -> int:
    from visionops.training import load_dataset_config, train_detector

    config = load_dataset_config(args.config)
    out = train_detector(
        config, args.out, args.seed, args.weights, args.epochs, args.imgsz, args.device
    )
    print(f"training_run={out}")
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    from visionops.evaluation import evaluate_detector
    from visionops.training import load_dataset_config

    print(f"metrics={evaluate_detector(load_dataset_config(args.config), args.model, args.out)}")
    return 0


def _run_export_onnx(args: argparse.Namespace) -> int:
    from visionops.export import export_onnx

    print(f"onnx={export_onnx(args.model, args.out, args.imgsz, args.sample_images)}")
    return 0


def _run_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from visionops.api import create_app

    uvicorn.run(create_app(args.model, args.max_upload_bytes), host=args.host, port=args.port)
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    import json

    from visionops.benchmark import benchmark_model
    from visionops.inference import OnnxPersonDetector

    detector = OnnxPersonDetector(args.model)
    report = benchmark_model(detector, args.images, args.warmup, args.iterations)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"benchmark={out}")
    return 0


def _run_make_showcase(args: argparse.Namespace) -> int:
    from visionops.inference import OnnxPersonDetector
    from visionops.showcase import make_showcase

    artifacts = make_showcase(
        args.video,
        OnnxPersonDetector(args.model),
        args.out,
        args.start_s,
        args.duration_s,
        source_url=args.source_url,
        license_name=args.license,
        attribution=args.attribution,
    )
    print(f"showcase={artifacts.video_path}")
    return 0


HANDLERS = {
    "ingest": _run_ingest,
    "prelabel": _run_prelabel,
    "select": _run_select,
    "export-labelstudio": _run_export_labelstudio,
    "train": _run_train,
    "evaluate": _run_evaluate,
    "export-onnx": _run_export_onnx,
    "serve": _run_serve,
    "benchmark": _run_benchmark,
    "make-showcase": _run_make_showcase,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)
