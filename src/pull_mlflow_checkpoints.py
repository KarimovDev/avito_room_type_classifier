from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.compare_mlflow_models import KNOWN_MODELS, load_best_rows
from src.training_helpers import PROJECT_ROOT, resolve_project_path, to_project_relative_path


DEFAULT_CHECKPOINTS: dict[tuple[str, str], str] = {
    ("resnet18", ""): "outputs/models/resnet18/resnet18_best.pt",
    ("resnet50", ""): "outputs/models/resnet50/resnet50_best.pt",
    ("densenet121", ""): "outputs/models/densenet121/densenet121_best.pt",
    ("efficientnet", "b0"): "outputs/models/efficientnet/efficientnet_b0_best.pt",
    ("efficientnet", "b1"): "outputs/models/efficientnet/efficientnet_b1_best.pt",
    ("convnext_nano", ""): "outputs/models/convnext_nano/convnext_nano_best.pt",
    ("convnext_tiny", ""): "outputs/models/convnext_tiny/convnext_tiny_best.pt",
    ("yolo", ""): "outputs/models/yolo/yolo_best.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download best MLflow checkpoints for known models (DagsHub remote by default)",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model names to download (default: all KNOWN_MODELS)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned downloads without fetching artifacts",
    )
    return parser.parse_args()


def _load_mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow не установлен. Выполните `just install-tracking`") from exc
    return mlflow


def _selected_models(raw: str) -> set[str] | None:
    if not raw.strip():
        return None
    models = {name.strip() for name in raw.split(",") if name.strip()}
    unknown = models - KNOWN_MODELS
    if unknown:
        raise ValueError(f"Unknown models: {', '.join(sorted(unknown))}")
    return models


def destination_path(row: dict[str, str]) -> Path | None:
    checkpoint = row.get("checkpoint", "").strip()
    if checkpoint:
        resolved = resolve_project_path(checkpoint)
        if resolved is not None:
            return resolved

    model = row["model"]
    variant = row.get("variant", "")
    default = DEFAULT_CHECKPOINTS.get((model, variant)) or DEFAULT_CHECKPOINTS.get((model, ""))
    if default is None:
        return None
    return PROJECT_ROOT / default


def _checkpoint_artifact_names(client: object, run_id: str) -> list[str]:
    names: list[str] = []
    for artifact in client.list_artifacts(run_id):
        if not artifact.is_dir and artifact.path.endswith(".pt"):
            names.append(Path(artifact.path).name)
    return names


def _pick_artifact_name(artifact_names: list[str], dest: Path) -> str | None:
    if dest.name in artifact_names:
        return dest.name
    if len(artifact_names) == 1:
        return artifact_names[0]
    return None


def download_checkpoint(mlflow: object, run_id: str, dest: Path, dry_run: bool) -> Path | None:
    client = mlflow.tracking.MlflowClient()
    artifact_names = _checkpoint_artifact_names(client, run_id)
    artifact_name = _pick_artifact_name(artifact_names, dest)
    if artifact_name is None:
        return None

    if dry_run:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    downloaded = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_name,
        dst_path=str(dest.parent),
    )
    downloaded_path = Path(downloaded)
    if downloaded_path.resolve() != dest.resolve():
        if downloaded_path.is_file():
            shutil.move(str(downloaded_path), dest)
        elif (dest.parent / artifact_name).exists():
            shutil.move(str(dest.parent / artifact_name), dest)
    return dest if dest.exists() else None


def main() -> int:
    args = parse_args()
    selected = _selected_models(args.models)

    mlflow = _load_mlflow()
    rows = load_best_rows()
    if selected is not None:
        rows = [row for row in rows if row["model"] in selected]

    if not rows:
        print("Нет подходящих MLflow run'ов. Проверьте авторизацию: just dagshub-login")
        return 1

    downloaded = 0
    skipped = 0
    for row in sorted(rows, key=lambda item: (item["model"], item.get("variant", ""))):
        dest = destination_path(row)
        if dest is None:
            print(f"skip model={row['model']} variant={row.get('variant', '')}: unknown destination")
            skipped += 1
            continue

        run_id = row["mlflow_run_id"]
        metric = row.get("best_macro_f1") or row.get("best_metric") or "n/a"
        rel_dest = to_project_relative_path(dest)
        if args.dry_run:
            print(f"dry-run run_id={run_id} metric={metric} -> {rel_dest}")
            downloaded += 1
            continue

        result = download_checkpoint(mlflow, run_id, dest, dry_run=False)
        if result is None:
            print(f"skip run_id={run_id} model={row['model']}: checkpoint artifact not found")
            skipped += 1
            continue

        print(f"downloaded run_id={run_id} model={row['model']} metric={metric} -> {rel_dest}")
        downloaded += 1

    print(f"downloaded={downloaded} skipped={skipped}")
    if downloaded == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
