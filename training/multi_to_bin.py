import argparse
import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
from tqdm import tqdm

from ocular_dataset import OcularDataset  # noqa: E402  # pyright: ignore[reportMissingImports]
from transforms import val_transform_sequence  # noqa: E402  # pyright: ignore[reportMissingImports]


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class Config:
    data_dir: str = "/workspace/data_15"
    csv_train: str = "/workspace/Retiniax/training_data/train_dataset.csv"
    csv_val: str = "/workspace/Retiniax/training_data/val_dataset.csv"
    checkpoint_path: str = "/workspace/data_15/best_dot_acc_0111_0120_ep0111_0.667376.pt"
    output_dir: str = "/workspace/data_15/binary_rf_head"
    model_name: str = "inception_next_small.sail_in1k"
    drop_rate: float = 0.0
    n_classes: int = 14
    batch_size: int = 32
    num_workers: int = 8
    seed: int = 42
    tune_xgboost: bool = False


class FeatureBackbone(nn.Module):
    """Same encoder + avg/max pooling used by training/script.py, without the multilabel head."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten(1)
        self.feature_size = self.encoder.num_features * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder.forward_features(x)
        x_avg = self.flatten(self.pool_avg(features))
        x_max = self.flatten(self.pool_max(features))
        return torch.cat([x_avg, x_max], dim=1)


class DeepClassifierForCheckpoint(nn.Module):
    """Matches the original trained model so its checkpoint can be loaded exactly."""

    def __init__(self, encoder: nn.Module, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten(1)
        self.feature_size = self.encoder.num_features * 2
        self.classifier = nn.Linear(self.feature_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder.forward_features(x)
        x_avg = self.flatten(self.pool_avg(features))
        x_max = self.flatten(self.pool_max(features))
        x_out = torch.cat([x_avg, x_max], dim=1)
        return self.classifier(x_out)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Train a binary RandomForest/XGBoost head on frozen features from the "
            "multilabel model trained by training/script.py."
        )
    )
    parser.add_argument("--data-dir", default=Config.data_dir)
    parser.add_argument("--csv-train", default=Config.csv_train)
    parser.add_argument("--csv-val", default=Config.csv_val)
    parser.add_argument("--checkpoint-path", default=Config.checkpoint_path)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--model-name", default=Config.model_name)
    parser.add_argument("--drop-rate", type=float, default=Config.drop_rate)
    parser.add_argument("--n-classes", type=int, default=Config.n_classes)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument(
        "--tune-xgboost",
        action="store_true",
        help="Also tune an XGBoost head if xgboost is installed.",
    )
    return Config(**vars(parser.parse_args()))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_targets_from_csv(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    if labels.shape[1] < 2:
        raise ValueError("Expected a multilabel CSV with NCS plus pathology columns.")
    # Column 0 is NCS/healthy. Binary class 1 means any non-NCS pathology is present.
    return (labels[:, 1:].sum(axis=1) > 0).astype(np.int64)


def validation_metadata_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    label_columns = df.columns[1:].tolist()
    pathology_columns = label_columns[1:]
    labels = df[label_columns].to_numpy(dtype=np.float32)

    original_classes = []
    for _, row in df[label_columns].iterrows():
        active_classes = [col for col in label_columns if row[col] > 0]
        original_classes.append(";".join(active_classes) if active_classes else "none")

    metadata = pd.DataFrame(
        {
            "image_name": df.iloc[:, 0].astype(str),
            "image_original_class": original_classes,
            "image_binary_class": (labels[:, 1:].sum(axis=1) > 0).astype(np.int64),
        }
    )
    for column in pathology_columns:
        metadata[column] = df[column].to_numpy(dtype=np.float32)
    return metadata


def build_loader(csv_path: str, data_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    dataset = OcularDataset(
        csv_file=csv_path,
        data_dir=data_dir,
        transform=val_transform_sequence,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )


def get_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise ValueError(
        "Could not find model weights in checkpoint. Expected a plain state_dict "
        "or a dict containing 'model_state_dict'."
    )


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def load_feature_backbone(config: Config, device: torch.device) -> FeatureBackbone:
    if not os.path.isfile(config.checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {config.checkpoint_path}. Pass --checkpoint-path to the trained .pt file."
        )

    encoder = timm.create_model(
        config.model_name,
        in_chans=3,
        pretrained=False,
        num_classes=0,
        drop_path_rate=config.drop_rate,
    )
    trained_model = DeepClassifierForCheckpoint(encoder=encoder, n_classes=config.n_classes)

    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    state_dict = strip_module_prefix(get_state_dict(checkpoint))
    missing, unexpected = trained_model.load_state_dict(state_dict, strict=False)
    unexpected_without_head = [key for key in unexpected if not key.startswith("classifier.")]
    if missing:
        print(f"Warning: missing checkpoint keys: {missing}")
    if unexpected_without_head:
        print(f"Warning: unexpected checkpoint keys: {unexpected_without_head}")

    feature_model = FeatureBackbone(trained_model.encoder)
    feature_model.to(device)
    feature_model.eval()
    for param in feature_model.parameters():
        param.requires_grad = False
    return feature_model


@torch.inference_mode()
def extract_features(
    feature_model: FeatureBackbone,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
) -> np.ndarray:
    features = []
    for batch in tqdm(loader, desc=f"Extracting {split_name} features"):
        images = batch["image"].to(device, non_blocking=True).float()
        batch_features = feature_model(images)
        features.append(batch_features.cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = None
    return metrics


def plot_pathology_prediction_by_original_class(
    metadata: pd.DataFrame,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    pathology_columns = [
        column
        for column in metadata.columns
        if column not in {"image_name", "image_original_class", "image_binary_class"}
    ]
    proportions = []
    counts = []
    for column in pathology_columns:
        mask = metadata[column].to_numpy(dtype=np.float32) > 0
        counts.append(int(mask.sum()))
        proportions.append(float(y_pred[mask].mean()) if mask.any() else 0.0)

    fig_width = max(12, len(pathology_columns) * 0.8)
    plt.figure(figsize=(fig_width, 6))
    bars = plt.bar(pathology_columns, proportions, color="#4472C4")
    plt.ylim(0, 1.0)
    plt.ylabel("Proportion predicted pathologic")
    plt.xlabel("Original pathology class")
    plt.title("Binary Model: Pathologic Predictions by Original Class")
    plt.xticks(rotation=45, ha="right")

    for bar, proportion, count in zip(bars, proportions, counts, strict=True):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            min(proportion + 0.02, 0.98),
            f"{proportion:.2f}\nn={count}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
    decision_threshold: float = 0.5,
) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    finite_thresholds = np.where(np.isfinite(thresholds), thresholds, np.nan)
    threshold_idx = int(np.nanargmin(np.abs(finite_thresholds - decision_threshold)))

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#4472C4", linewidth=2, label=f"ROC curve (AUC = {auc_value:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random")
    plt.scatter(
        fpr[threshold_idx],
        tpr[threshold_idx],
        color="#C00000",
        zorder=3,
        label=f"Threshold ~= {thresholds[threshold_idx]:.3f}",
    )
    plt.text(
        0.60,
        0.08,
        f"AUC = {auc_value:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.9},
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Best Binary Head ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return float(auc_value)


def save_prediction_csv(
    metadata: pd.DataFrame,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    predictions = metadata[["image_name", "image_original_class", "image_binary_class"]].copy()
    predictions["probability_healthy"] = 1.0 - y_prob
    predictions["probability_pathologic"] = y_prob
    predictions["predicted_binary_class"] = y_pred
    predictions.to_csv(output_path, index=False)


def tune_random_forest(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[RandomForestClassifier, dict[str, Any], list[dict[str, Any]]]:
    param_grid = {
        "n_estimators": [300, 600, 1000],
        "max_depth": [None, 8, 16, 32],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    best_model: RandomForestClassifier | None = None
    best_result: dict[str, Any] | None = None
    all_results = []

    for params in tqdm(list(ParameterGrid(param_grid)), desc="Tuning RandomForest"):
        model = RandomForestClassifier(
            **params,
            random_state=seed,
            n_jobs=-1,
            bootstrap=True,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        y_prob = model.predict_proba(x_val)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_prob)
        result = {"model_type": "random_forest", "params": params, "metrics": metrics}
        all_results.append(result)

        if best_result is None or metrics["f1"] > best_result["metrics"]["f1"]:
            best_result = result
            best_model = model
            print(
                "New best RF | "
                f"F1={metrics['f1']:.4f} | "
                f"BalancedAcc={metrics['balanced_accuracy']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"Params={params}"
            )

    if best_model is None or best_result is None:
        raise RuntimeError("RandomForest tuning did not produce a model.")
    return best_model, best_result, all_results


def tune_xgboost_if_available(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[Any | None, dict[str, Any] | None, list[dict[str, Any]]]:
    try:
        from xgboost import XGBClassifier  # pyright: ignore[reportMissingImports]
    except ImportError:
        print("xgboost is not installed; skipping XGBoost tuning.")
        return None, None, []

    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)
    param_grid = {
        "n_estimators": [300, 600],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [1.0, 5.0],
    }

    best_model = None
    best_result: dict[str, Any] | None = None
    all_results = []

    for params in tqdm(list(ParameterGrid(param_grid)), desc="Tuning XGBoost"):
        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        y_prob = model.predict_proba(x_val)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_prob)
        result = {"model_type": "xgboost", "params": params, "metrics": metrics}
        all_results.append(result)

        if best_result is None or metrics["f1"] > best_result["metrics"]["f1"]:
            best_result = result
            best_model = model
            print(
                "New best XGB | "
                f"F1={metrics['f1']:.4f} | "
                f"BalancedAcc={metrics['balanced_accuracy']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"Params={params}"
            )

    return best_model, best_result, all_results


def save_artifacts(
    output_dir: str,
    config: Config,
    best_model: Any,
    best_result: dict[str, Any],
    all_results: list[dict[str, Any]],
    y_val: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    val_metadata: pd.DataFrame,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "best_binary_head.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open(output_path / "best_result.json", "w", encoding="utf-8") as f:
        json.dump(best_result, f, indent=2)

    with open(output_path / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    with open(output_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)

    report = classification_report(
        y_val,
        y_pred,
        target_names=["class_0_ncs", "class_1_pathology"],
        zero_division=0,
    )
    with open(output_path / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    plot_pathology_prediction_by_original_class(
        metadata=val_metadata,
        y_pred=y_pred,
        output_path=output_path / "pathologic_prediction_by_original_class.png",
    )
    auc_value = plot_roc_auc(
        y_true=y_val,
        y_prob=y_prob,
        output_path=output_path / "roc_auc_best_binary_head.png",
    )
    best_result["metrics"]["roc_auc"] = auc_value

    save_prediction_csv(
        metadata=val_metadata,
        y_prob=y_prob,
        y_pred=y_pred,
        output_path=output_path / "validation_predictions.csv",
    )


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = build_loader(
        config.csv_train,
        config.data_dir,
        config.batch_size,
        config.num_workers,
    )
    val_loader = build_loader(
        config.csv_val,
        config.data_dir,
        config.batch_size,
        config.num_workers,
    )

    y_train = binary_targets_from_csv(config.csv_train)
    y_val = binary_targets_from_csv(config.csv_val)
    val_metadata = validation_metadata_from_csv(config.csv_val)
    print(
        "Binary label distribution | "
        f"train: class0={(y_train == 0).sum()}, class1={(y_train == 1).sum()} | "
        f"val: class0={(y_val == 0).sum()}, class1={(y_val == 1).sum()}"
    )

    feature_model = load_feature_backbone(config, device)
    x_train = extract_features(feature_model, train_loader, device, "train")
    x_val = extract_features(feature_model, val_loader, device, "val")
    print(f"Feature shapes | train: {x_train.shape}, val: {x_val.shape}")

    rf_model, rf_best, rf_results = tune_random_forest(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        seed=config.seed,
    )
    candidates = [(rf_model, rf_best)]
    all_results = rf_results

    if config.tune_xgboost:
        xgb_model, xgb_best, xgb_results = tune_xgboost_if_available(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            seed=config.seed,
        )
        all_results.extend(xgb_results)
        if xgb_model is not None and xgb_best is not None:
            candidates.append((xgb_model, xgb_best))

    best_model, best_result = max(candidates, key=lambda item: item[1]["metrics"]["f1"])
    y_pred = best_model.predict(x_val)
    y_prob = best_model.predict_proba(x_val)[:, 1]
    best_result["metrics"] = compute_metrics(y_val, y_pred, y_prob)

    print("\nBest binary head")
    print(json.dumps(best_result, indent=2))
    print("\nValidation classification report")
    print(
        classification_report(
            y_val,
            y_pred,
            target_names=["class_0_ncs", "class_1_pathology"],
            zero_division=0,
        )
    )

    save_artifacts(
        output_dir=config.output_dir,
        config=config,
        best_model=best_model,
        best_result=best_result,
        all_results=all_results,
        y_val=y_val,
        y_pred=y_pred,
        y_prob=y_prob,
        val_metadata=val_metadata,
    )
    print(f"Saved binary head and reports to: {config.output_dir}")


if __name__ == "__main__":
    main()
