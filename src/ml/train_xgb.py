"""
Train XGBoost model for BTC price prediction.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from src.core.config import settings
from src.ml.features import build_ml_dataset
from src.services.ohlcv_service import load_ohlcv_df


def train_xgb_model(
    horizon: int = 5,
    train_split: float = 0.8,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    use_events: bool | None = None,
) -> XGBClassifier:
    """
    Train XGBoost classifier for BTC price direction prediction.

    Args:
        horizon: Number of periods ahead to predict
        train_split: Proportion of data to use for training (rest for validation)
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        subsample: Subsample ratio of training instances
        colsample_bytree: Subsample ratio of columns when constructing each tree

    Returns:
        Trained XGBClassifier model
    """
    print("Loading OHLCV data...")
    df = load_ohlcv_df()

    if use_events is None:
        use_events = settings.EVENTS_ENABLED
    print(
        f"Building ML dataset (horizon={horizon}, use_events={use_events})..."
    )
    X, y = build_ml_dataset(
        df,
        horizon=horizon,
        use_events=use_events,
    )

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Positive samples: {y.sum()} ({y.mean()*100:.2f}%)")

    # Time-series split (80% train, 20% validation)
    split_idx = int(len(X) * train_split)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train: {len(X_train)}, Validation: {len(X_val)}")

    # Train XGBoost model
    print("Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Feature importance
    feature_importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    print("\nTop 10 Features by Importance:")
    print(feature_importance.head(10).to_string(index=False))

    # Save model
    model_path = Path(settings.XGB_MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving model to {model_path}...")
    joblib.dump(model, model_path)
    print("✅ Model saved successfully!")

    return model


if __name__ == "__main__":
    train_xgb_model()

