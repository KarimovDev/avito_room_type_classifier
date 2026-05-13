"""Обучение ConvNeXt-Tiny: метрики и отчёты как у ResNet18. Настройки — в train_config.json рядом со скриптом."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import amp, nn
from torch.nn import functional as F
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.convnext_tiny.model import build_convnext_tiny
from src.dataloaders import create_dataloaders
from src.device import get_default_device
from src.labels import load_label_mapping
from src.metrics import calculate_accuracy, calculate_macro_f1, calculate_per_class_f1
from src.training_helpers import build_checkpoint, to_project_relative_path

_CONFIG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _CONFIG_DIR / "train_config.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ConvNeXt-Tiny (настройки в JSON)")
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"JSON с гиперпараметрами и путями (по умолчанию: {_DEFAULT_CONFIG.name} в этой папке).",
    )
    return p.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Нет файла конфигурации: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _p(root: Path, rel: str | None) -> Path | None:
    if rel is None or rel == "":
        return None
    q = Path(rel)
    return q if q.is_absolute() else (root / q).resolve()


def _resolve_csv(csv_path: Path, default_file: str) -> Path:
    p = csv_path.expanduser()
    try:
        p = p.resolve()
    except OSError:
        p = csv_path
    if p.exists() and p.is_dir():
        out = p / default_file
        print(f"CSV: каталог {csv_path} -> файл {out}", flush=True)
        return out
    return csv_path


def _bool(v: Any, default: bool) -> bool:
    if v is None:
        return default
    return bool(v)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_csv_exclude_class(src: Path, dst: Path, exclude_id: int) -> tuple[int, int]:
    df = pd.read_csv(src)
    if "result" not in df.columns:
        raise ValueError(f"Нет колонки result: {src}")
    r = df["result"].astype(int)
    dropped = int((r == exclude_id).sum())
    df = df.loc[r != exclude_id].copy()
    r2 = df["result"].astype(int)
    df["result"] = np.where(r2 > exclude_id, r2 - 1, r2)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    return len(df), dropped


def validate_paths(train_csv: Path, val_csv: Path, train_img: Path, val_img: Path) -> None:
    for label, p in (("--train-csv", train_csv), ("--val-csv", val_csv)):
        if not p.is_file():
            raise FileNotFoundError(f"{label}: ожидается файл CSV: {p}")
    for label, p in (("--train-images", train_img), ("--val-images", val_img)):
        if not p.exists():
            raise FileNotFoundError(f"{label}: не найдено: {p}")


def get_class_weights(csv_path: Path, num_classes: int, device: torch.device) -> torch.Tensor:
    t = torch.tensor(pd.read_csv(csv_path)["result"].astype(int).to_list(), dtype=torch.int64)
    if int(t.min()) < 0:
        raise ValueError("Метки result не могут быть отрицательными")
    need = int(t.max().item()) + 1
    if need > num_classes:
        raise ValueError(f"В train до метки {int(t.max())}, нужно num_classes >= {need}")
    counts = torch.bincount(t, minlength=num_classes).float()
    w = torch.zeros(num_classes, dtype=torch.float32)
    ok = counts > 0
    w[ok] = counts.sum() / (ok.sum() * counts[ok])
    return w.to(device)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    device: torch.device,
    *,
    use_amp: bool,
    show_progress: bool,
    epoch: int,
) -> float:
    model.train()
    amp_on = use_amp and device.type == "cuda"
    it = tqdm(loader, desc=f"train ep{epoch}", leave=False) if show_progress else loader
    total = 0.0
    n = len(loader.dataset)
    for images, targets in it:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast("cuda", enabled=amp_on):
            logits = model(images)
            loss = criterion(logits, targets)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item() * images.size(0)
    return total / max(n, 1)


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    *,
    use_amp: bool,
    show_progress: bool,
    epoch: int,
) -> tuple[float, float, float, list[dict[str, object]]]:
    model.eval()
    amp_on = use_amp and device.type == "cuda"
    total_loss = 0.0
    cls_sum = torch.zeros(num_classes, dtype=torch.float64)
    cls_cnt = torch.zeros(num_classes, dtype=torch.float64)
    y_true: list[int] = []
    y_pred: list[int] = []
    it = tqdm(loader, desc=f"val ep{epoch}", leave=False) if show_progress else loader
    w = criterion.weight
    for images, targets in it:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with amp.autocast("cuda", enabled=amp_on):
            logits = model(images)
            loss = criterion(logits, targets)
            psl = F.cross_entropy(logits, targets, weight=w, reduction="none")
        pred = logits.argmax(dim=1)
        bs = images.size(0)
        total_loss += loss.item() * bs
        for c in range(num_classes):
            m = targets == c
            if m.any():
                cls_sum[c] += psl[m].sum().double().cpu()
                cls_cnt[c] += m.sum().cpu()
        y_true.extend(targets.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    n = len(loader.dataset)
    acc = float(calculate_accuracy(y_true, y_pred))
    macro = float(calculate_macro_f1(y_true, y_pred))
    per = calculate_per_class_f1(y_true, y_pred, num_classes)
    cls_loss = cls_sum / cls_cnt.clamp_min(1)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    for item in per:
        cid = int(item["class_id"])
        m = yt == cid
        item["accuracy"] = float((yp[m] == cid).mean()) if m.any() else 0.0
        item["loss"] = float(cls_loss[cid])
    return total_loss / max(n, 1), acc, macro, per


def _orig_class(mid: int, excluded: int | None) -> int:
    if excluded is None:
        return mid
    return mid if mid < excluded else mid + 1


def add_label_names(per: list[dict[str, object]], excluded: int | None) -> list[dict[str, object]]:
    lm = load_label_mapping()
    out = []
    for item in per:
        oid = _orig_class(int(item["class_id"]), excluded)
        row = {**item, "label": lm.get(oid, str(oid))}
        if excluded is not None:
            row["original_class_id"] = oid
        out.append(row)
    return out


def _atomic_torch_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _load_ckpt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_ckpt_payload(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: amp.GradScaler,
    *,
    meta: dict[str, Any],
    completed_epoch: int,
    macro_f1: float,
    run_id: str,
    best_f1: float,
    best_ep: int,
    no_improve: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        **build_checkpoint(
            model=model,
            model_name=str(meta.get("model_name", "convnext_tiny")),
            epoch=completed_epoch,
            best_metric=best_f1 if math.isfinite(best_f1) else macro_f1,
            optimizer=opt,
            checkpoint_path=meta.get("checkpoint_path"),
        ),
        "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
        "completed_epoch": completed_epoch,
        "macro_f1": float(macro_f1),
        "best_macro_f1": float(best_f1) if math.isfinite(best_f1) else None,
        "best_epoch": int(best_ep),
        "epochs_without_improvement": int(no_improve),
        "run_id": run_id,
        **meta,
    }
    return out


def save_metrics_report(metrics: dict[str, Any], metrics_dir: Path, model_name: str) -> tuple[Path, Path]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rid = str(metrics.get("run_id") or "")
    blob = {"run_id": rid, **metrics}
    mp = metrics_dir / f"{model_name}_metrics.json"
    ep = metrics_dir / f"{model_name}_experiments.json"
    mp.write_text(json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")
    hp = metrics["hyperparameters"]
    be = metrics["best_epoch_metrics"]
    exp = {
        "run_id": rid,
        "model": metrics["model"],
        "model_name": model_name,
        "metrics_dir": to_project_relative_path(metrics_dir),
        "best_epoch": metrics["best_epoch"],
        "best_macro_f1": metrics["best_macro_f1"],
        "best_accuracy": be.get("accuracy"),
        "best_train_loss": be.get("train_loss"),
        "best_val_loss": be.get("val_loss"),
        "stop_reason": metrics["stop_reason"],
        "checkpoint": metrics["checkpoint"],
        "epochs": hp["epochs"],
        "batch_size": hp["batch_size"],
        "image_size": hp["image_size"],
        "learning_rate": hp["learning_rate"],
        "weight_decay": hp["weight_decay"],
        "seed": hp["seed"],
        "metrics_json": str(mp),
    }
    for k in ("exclude_class_id", "amp", "weighted_sampling"):
        if k in hp:
            exp[k] = hp[k]
    hist = json.loads(ep.read_text(encoding="utf-8")) if ep.exists() else []
    hist.append(exp)
    ep.write_text(json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
    return mp, ep


def append_history(
    metrics_dir: Path,
    model_name: str,
    run_id: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    accuracy: float,
    macro_f1: float,
    best_f1: float,
    best_ep: int,
) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    hp = metrics_dir / f"{model_name}_epoch_history.jsonl"
    line = json.dumps(
        {
            "run_id": run_id,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "best_macro_f1_so_far": best_f1,
            "best_epoch_so_far": best_ep,
        },
        ensure_ascii=False,
    )
    with hp.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _f1_improved(cur: float, best: float, delta: float) -> bool:
    if not math.isfinite(cur):
        return False
    if not math.isfinite(best):
        return True
    return cur > best + delta


def _log_torch_and_device(device: torch.device) -> None:
    """Печатает версию torch и доступность CUDA (uv/pip без cu-индекса дают CPU-сборку)."""
    print(f"torch {torch.__version__}", flush=True)
    print(f"torch.version.cuda (сборка) = {torch.version.cuda}", flush=True)
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"выбранное устройство = {device}", flush=True)
    if device.type != "cuda" and not torch.cuda.is_available():
        print(
            "Подсказка: `uv sync --group data` обычно ставит CPU-only torch с PyPI. "
            "Поставьте CUDA-сборку в тот же venv, из корня room_type_classifier:\n"
            "  uv pip install torch==2.5.1 torchvision==0.20.1 "
            "--extra-index-url https://download.pytorch.org/whl/cu124\n"
            "Другая версия CUDA у драйвера - см. https://pytorch.org/get-started/locally/ (например cu121: .../whl/cu121)",
            flush=True,
        )


def main() -> None:
    args = _parse_args()
    cfg = _load_json(args.config)

    model_name = str(cfg.get("model_name", "convnext_tiny"))
    num_classes = int(cfg.get("num_classes", 20))
    epochs = int(cfg.get("epochs", 30))
    batch_size = int(cfg.get("batch_size", 32))
    num_workers = int(cfg.get("num_workers", 0))
    image_size = int(cfg.get("image_size", 224))
    lr = float(cfg.get("learning_rate", 1e-4))
    wd = float(cfg.get("weight_decay", 1e-4))
    seed = int(cfg.get("seed", 42))
    exclude_id = int(cfg.get("exclude_class_id", 18))
    excluded = None if exclude_id < 0 else exclude_id

    tc = cfg.get("train_csv")
    vc = cfg.get("val_csv")
    if not tc or not vc:
        raise ValueError("В train_config.json нужны непустые train_csv и val_csv")
    train_csv = _resolve_csv(_p(ROOT_DIR, str(tc)), "train_df.csv")
    val_csv = _resolve_csv(_p(ROOT_DIR, str(vc)), "val_df.csv")
    ti = cfg.get("train_images")
    vi = cfg.get("val_images")
    train_img = _p(ROOT_DIR, str(ti)) if ti else ROOT_DIR / "data/raw/train_images"
    val_img = _p(ROOT_DIR, str(vi)) if vi else ROOT_DIR / "data/raw/val_images"
    validate_paths(train_csv, val_csv, train_img, val_img)

    train_eff, val_eff = train_csv, val_csv
    if excluded is not None:
        td = Path(tempfile.mkdtemp(prefix="rtc_cnx_"))
        train_eff, val_eff = td / "tr.csv", td / "va.csv"
        nt, dt = filter_csv_exclude_class(train_csv, train_eff, excluded)
        nv, dv = filter_csv_exclude_class(val_csv, val_eff, excluded)
        if nt == 0 or nv == 0:
            raise ValueError(f"После исключения класса {excluded} train={nt}, val={nv}")
        inferred = int(pd.read_csv(train_eff)["result"].max()) + 1
        if num_classes != inferred:
            print(f"num_classes {num_classes} -> {inferred} (по train после exclude)", flush=True)
            num_classes = inferred
        print(f"Исключён класс {excluded}: train {nt} (−{dt}), val {nv} (−{dv})", flush=True)

    out_dir = (_p(ROOT_DIR, cfg.get("output_dir")) or (ROOT_DIR / "outputs/models" / model_name)).resolve()
    met_dir = (_p(ROOT_DIR, cfg.get("metrics_dir")) or (ROOT_DIR / "reports/metrics" / model_name)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    set_seed(seed)

    device = get_default_device()
    _log_torch_and_device(device)
    if _bool(cfg.get("require_cuda"), False) and device.type != "cuda":
        raise RuntimeError(
            "В train_config.json включено require_cuda, но CUDA недоступна. "
            "Переустановите torch/torchvision с CUDA (см. сообщение выше) или поставьте require_cuda: false."
        )

    pretrained = _bool(cfg.get("pretrained"), True)
    use_class_w = _bool(cfg.get("class_weights"), True)
    use_wsample = _bool(cfg.get("weighted_sampling"), False)
    save_ckpt = _bool(cfg.get("save_checkpoint"), True)
    save_last = _bool(cfg.get("save_last_every_epoch"), True)
    use_amp = _bool(cfg.get("amp"), True) and device.type == "cuda"
    show_progress = _bool(cfg.get("show_progress"), True)
    es_pat = int(cfg.get("early_stopping_patience", 3))
    es_delta = float(cfg.get("early_stopping_min_delta", 1e-4))

    pin_m = cfg.get("pin_memory")
    pers_w = cfg.get("persistent_workers")
    if device.type == "cuda" and sys.platform == "win32":
        pin_b = _bool(pin_m, False)
        pers_b = _bool(pers_w, False) and num_workers > 0
        if not pin_b:
            print("Windows+CUDA: pin_memory=False (override в JSON: pin_memory)", flush=True)
    elif device.type == "cuda":
        pin_b = True if pin_m is None else _bool(pin_m, True)
        pers_b = (num_workers > 0) if pers_w is None else (_bool(pers_w, False) and num_workers > 0)
    else:
        pin_b, pers_b = False, False

    if _bool(cfg.get("cudnn_benchmark"), True) and device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"device={device} run_id={run_id}", flush=True)
    print(f"config={args.config.resolve()}", flush=True)
    print(f"output_dir={out_dir} metrics_dir={met_dir}", flush=True)

    train_loader, val_loader = create_dataloaders(
        train_csv_path=train_eff,
        val_csv_path=val_eff,
        train_image_root=str(train_img),
        val_image_root=str(val_img),
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        use_weighted_sampling=use_wsample,
        seed=seed,
        pin_memory=pin_b,
        persistent_workers=pers_b,
    )
    print(
        f"train={len(train_loader.dataset)} ({len(train_loader)} батчей) val={len(val_loader.dataset)}",
        flush=True,
    )
    try:
        if "image_path" in pd.read_csv(train_eff, nrows=1).columns:
            print(
                "Данные: CSV после preprocess_data (есть image_path). "
                "train_images/val_images в JSON - каталоги с jpg (обычно data/raw/..._images)",
                flush=True,
            )
    except Exception:
        pass

    resume_raw = cfg.get("resume_from")
    resume_path = _p(ROOT_DIR, str(resume_raw)) if resume_raw else None
    use_pretrained = False if resume_path and resume_path.is_file() else pretrained

    model = build_convnext_tiny(num_classes=num_classes, pretrained=use_pretrained).to(device)
    cw = get_class_weights(train_eff, num_classes, device) if use_class_w else None
    crit = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scaler = amp.GradScaler("cuda", enabled=use_amp)

    if use_amp:
        print("AMP включён", flush=True)

    best_path = out_dir / f"{model_name}_best.pt"
    last_path = out_dir / f"{model_name}_last.pt"
    best_f1 = float("-inf")
    best_ep = 0
    best_metrics: dict[str, Any] = {}
    no_improve = 0
    stop = "max_epochs"
    ckpt_meta = {
        "num_classes": num_classes,
        "image_size": image_size,
        "excluded_original_class_id": excluded,
        "config_path": to_project_relative_path(args.config),
        "model_name": model_name,
        "idx_to_class": {str(class_id): label for class_id, label in load_label_mapping().items()},
    }

    start_ep = 1
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume_from: файл не найден: {resume_path}")
        ck = _load_ckpt(resume_path)
        if int(ck.get("num_classes", num_classes)) != num_classes:
            raise ValueError(
                f"num_classes в чекпоинте ({ck.get('num_classes')}) != текущему ({num_classes})"
            )
        model.load_state_dict(ck["model_state_dict"], strict=True)
        if ck.get("optimizer_state_dict"):
            try:
                opt.load_state_dict(ck["optimizer_state_dict"])
                print("Загружен optimizer_state_dict", flush=True)
            except Exception as e:
                print(f"optimizer_state_dict не загружен ({e}), новый AdamW", flush=True)
        sd = ck.get("scaler_state_dict")
        if sd is not None and scaler.is_enabled():
            try:
                scaler.load_state_dict(sd)
                print("Загружен scaler_state_dict", flush=True)
            except Exception as e:
                print(f"scaler не загружен ({e})", flush=True)
        done = int(ck.get("completed_epoch", ck.get("epoch", 0)))
        start_ep = done + 1
        bf = ck.get("best_macro_f1")
        if bf is not None and math.isfinite(float(bf)):
            best_f1 = float(bf)
        be = ck.get("best_epoch")
        if be is not None:
            best_ep = int(be)
        ni = ck.get("epochs_without_improvement")
        if ni is not None:
            no_improve = int(ni)
        print(
            f"Продолжение с {resume_path}: completed_epoch={done} -> старт с эпохи {start_ep}, "
            f"best_macro_f1={best_f1} best_epoch={best_ep} no_improve={no_improve}",
            flush=True,
        )

    if start_ep > epochs:
        print(f"Нечего делать: start_ep={start_ep} > epochs={epochs} (увеличь epochs в JSON)", flush=True)
        return

    for ep in range(start_ep, epochs + 1):
        t0 = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"--- epoch {ep}/{epochs} start {t0} ---", flush=True)
        tl = train_one_epoch(model, train_loader, crit, opt, scaler, device, use_amp=use_amp, show_progress=show_progress, epoch=ep)
        vl, acc, macro, per = validate(
            model, val_loader, crit, device, num_classes, use_amp=use_amp, show_progress=show_progress, epoch=ep
        )
        if not math.isfinite(macro):
            print("macro_f1 не число - подставляем 0.0 (иначе чекпоинт не сохранится)", flush=True)
            macro = 0.0
        per = add_label_names(per, excluded)

        improved = _f1_improved(macro, best_f1, es_delta)
        if improved:
            best_f1 = macro
            best_ep = ep
            no_improve = 0
            best_metrics = {
                "epoch": ep,
                "train_loss": tl,
                "val_loss": vl,
                "accuracy": acc,
                "macro_f1": macro,
                "per_class_metrics": [
                    {
                        "class_id": x["class_id"],
                        "label": x["label"],
                        "f1": x["f1"],
                        "accuracy": x["accuracy"],
                        "loss": x["loss"],
                        "support": x["support"],
                    }
                    for x in per
                ],
            }
            if save_ckpt:
                payload = _build_ckpt_payload(
                    model,
                    opt,
                    scaler,
                    meta={**ckpt_meta, "checkpoint_path": best_path},
                    completed_epoch=ep,
                    macro_f1=macro,
                    run_id=run_id,
                    best_f1=best_f1,
                    best_ep=best_ep,
                    no_improve=no_improve,
                )
                _atomic_torch_save(payload, best_path)
                print(f"checkpoint best -> {best_path} macro_f1={macro:.4f}", flush=True)
        else:
            no_improve += 1

        if save_ckpt and save_last:
            _atomic_torch_save(
                _build_ckpt_payload(
                    model,
                    opt,
                    scaler,
                    meta={**ckpt_meta, "checkpoint_path": last_path},
                    completed_epoch=ep,
                    macro_f1=macro,
                    run_id=run_id,
                    best_f1=best_f1,
                    best_ep=best_ep,
                    no_improve=no_improve,
                ),
                last_path,
            )

        print(
            f"epoch={ep} train_loss={tl:.4f} val_loss={vl:.4f} acc={acc:.4f} macro_f1={macro:.4f} "
            f"best={best_f1:.4f} best_ep={best_ep} no_improve={no_improve}",
            flush=True,
        )
        append_history(met_dir, model_name, run_id, ep, tl, vl, acc, macro, best_f1, best_ep)

        if es_pat > 0 and no_improve >= es_pat:
            stop = "early_stopping"
            print(f"early stopping после {es_pat} эпох без роста macro_f1", flush=True)
            break

    hp = {
        "model_name": model_name,
        "config": to_project_relative_path(args.config),
        "metrics_dir": to_project_relative_path(met_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "learning_rate": lr,
        "weight_decay": wd,
        "seed": seed,
        "pretrained": pretrained,
        "class_weights": use_class_w,
        "weighted_sampling": use_wsample,
        "save_checkpoint": save_ckpt,
        "save_last_every_epoch": save_last,
        "early_stopping_patience": es_pat,
        "early_stopping_min_delta": es_delta,
        "amp": use_amp,
        "show_progress": show_progress,
        "pin_memory": pin_b,
        "persistent_workers": pers_b,
        "exclude_class_id": excluded,
        "resume_from": to_project_relative_path(resume_path),
        "require_cuda": _bool(cfg.get("require_cuda"), False),
    }
    metrics = {
        "run_id": run_id,
        "model": model_name,
        "hyperparameters": hp,
        "best_epoch": best_ep,
        "best_macro_f1": float(best_f1) if math.isfinite(best_f1) else None,
        "best_epoch_metrics": best_metrics,
        "checkpoint": None if not save_ckpt else to_project_relative_path(best_path),
        "last_checkpoint": None if not save_ckpt or not save_last else to_project_relative_path(last_path),
        "stop_reason": stop,
    }
    mp, ep = save_metrics_report(metrics, met_dir, model_name)
    print(f"metrics -> {mp}", flush=True)
    print(f"experiments -> {ep}", flush=True)
    if save_ckpt:
        print(f"best.pt exists={best_path.is_file()} last.pt exists={last_path.is_file()}", flush=True)


if __name__ == "__main__":
    main()
