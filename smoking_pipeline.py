"""
Pipeline для датасета Body Signal of Smoking.
В одном классе собраны:
- предобработка данных;
- обучение нейронной сети;
- сохранение и загрузка модели/параметров предобработки;
- использование модели для predict / predict_proba.
"""

from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Union, List

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

@dataclass
class SmokingPipelineConfig:
    target_col: str = "smoking"
    test_size: float = 0.25
    random_state: int = 9
    batch_size: int = 128
    threshold: float = 0.5
    learning_rate: float = 0.001
    loss_name: str = "binary_focal_crossentropy"


class SmokingNNPipeline:
    """
    Единый pipeline:
    1. preprocess / fit_preprocess;
    2. train;
    3. evaluate;
    4. find_best_threshold;
    5. predict / predict_proba;
    6. save / load.
    """

    def __init__(self, config: Optional[SmokingPipelineConfig] = None):
        self.config = config or SmokingPipelineConfig()
        self.model: Optional[tf.keras.Model] = None
        self.scalers: Dict[str, object] = {}
        self.feature_columns: Optional[List[str]] = None
        self.history = None
        self.log_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # 1. Базовая подготовка колонок
    # ------------------------------------------------------------------
    def _prepare_base_dataframe(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Кодирует категориальные признаки и удаляет лишние колонки."""
        data = df.copy()
        
        for column in ["oral", "ID", "hearing(left)", "hearing(right)", "Cholesterol", "LDL", "Urine protein", "AST"]:
            if column in data.columns:
                data = data.drop(columns=column)

        # gender: M -> 1, F -> 0. Если уже числовой/boolean, просто приводим к int.
        if "gender" in data.columns:
            if data["gender"].dtype == "object" or data["gender"].dtype == "str":
                data["gender"] = data["gender"].map({"F": 0, "M": 1}).fillna(data["gender"])
            data["gender"] = data["gender"].astype(int)
        
        # tartar: Y -> 1, N -> 0.
        if "tartar" in data.columns:
            if data["tartar"].dtype == "object" or data["tartar"].dtype == "str":
                data["tartar"] = data["tartar"].map({"N": 0, "Y": 1}).fillna(data["tartar"])
            data["tartar"] = data["tartar"].astype(int)

        # dental caries: Y -> 1, N -> 0.
        if "dental caries" in data.columns:
            if data["dental caries"].dtype == "object" or data["dental caries"].dtype == "str":
                data["dental caries"] = data["dental caries"].map({"N": 0, "Y": 1}).fillna(data["dental caries"])
            data["dental caries"] = data["dental caries"].astype(int)

        return data

    def _filter_rows(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        condition: pd.Series,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        X = X.loc[condition].copy()
        if y is not None:
            y = y.loc[condition].copy()
        return X, y

    def _fit_and_scale(
        self,
        X: pd.DataFrame,
        col: str,
        scaler,
        use_log1p: bool = False,
    ) -> pd.DataFrame:
        values = X[[col]]
        if use_log1p:
            values = np.log1p(values)
        fitted_scaler = scaler.fit(values)
        self.scalers[col] = {"scaler": fitted_scaler, "use_log1p": use_log1p}
        X[col] = fitted_scaler.transform(values).astype(np.float32)
        return X

    def _transform_scale(self, X: pd.DataFrame, col: str) -> pd.DataFrame:
        params = self.scalers[col]
        values = X[[col]]
        if params["use_log1p"]:
            values = np.log1p(values)
        X[col] = params["scaler"].transform(values).astype(np.float32)
        return X

    # ------------------------------------------------------------------
    # 2. Предобработка train/test
    # ------------------------------------------------------------------
    def _fit_transform_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], pd.Index]:
        """
        Предобработка train-части.
        Повторяет логику из ноутбука: фильтры выбросов + scalers.
        """
        X = X.copy()
        y = y.copy() if y is not None else None

        # Логика из исходника.
        X = self._fit_and_scale(X, "age", RobustScaler(), use_log1p=True)
        X = self._fit_and_scale(X, "height(cm)", RobustScaler(), use_log1p=False)
        X = self._fit_and_scale(X, "weight(kg)", RobustScaler(), use_log1p=True)
        X = self._fit_and_scale(X, "waist(cm)", RobustScaler(), use_log1p=False)

        X, y = self._filter_rows(X, y, X["eyesight(left)"] <= 2)
        X["eyesight(left)"] = X["eyesight(left)"] - 1

        X, y = self._filter_rows(X, y, X["eyesight(right)"] <= 2)
        X["eyesight(right)"] = X["eyesight(right)"] - 1

        X, y = self._filter_rows(X, y, X["systolic"] <= 180)
        X = self._fit_and_scale(X, "systolic", StandardScaler(), use_log1p=False)

        X, y = self._filter_rows(X, y, X["relaxation"] <= 120)
        X = self._fit_and_scale(X, "relaxation", StandardScaler(), use_log1p=False)

        X, y = self._filter_rows(X, y, X["fasting blood sugar"] <= 350)
        X = self._fit_and_scale(X, "fasting blood sugar", RobustScaler(), use_log1p=True)

        X = self._fit_and_scale(X, "triglyceride", RobustScaler(), use_log1p=True)

        X, y = self._filter_rows(X, y, X["HDL"] <= 300)
        X = self._fit_and_scale(X, "HDL", RobustScaler(), use_log1p=False)

        X = self._fit_and_scale(X, "hemoglobin", RobustScaler(), use_log1p=False)

        X, y = self._filter_rows(X, y, X["ALT"] <= 100)
        X = self._fit_and_scale(X, "ALT", RobustScaler(), use_log1p=False)

        X = self._fit_and_scale(X, "Gtp", RobustScaler(), use_log1p=True)

        self.feature_columns = list(X.columns)
        X_array = X.astype(np.float32).to_numpy()
        y_array = y.astype(np.float32).to_numpy() if y is not None else None
        return X_array, y_array, X.index

    def _transform_features(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], pd.Index]:
        """
        Предобработка validation/test/inference.
        """
        if self.feature_columns is None:
            raise RuntimeError("Pipeline ещё не обучен: feature_columns отсутствуют.")

        X = X.copy()
        y = y.copy() if y is not None else None

        X = self._transform_scale(X, "age")
        X = self._transform_scale(X, "height(cm)")
        X = self._transform_scale(X, "weight(kg)")
        X = self._transform_scale(X, "waist(cm)")

        X, y = self._filter_rows(X, y, X["eyesight(left)"] <= 2)
        X["eyesight(left)"] = X["eyesight(left)"] - 1

        X, y = self._filter_rows(X, y, X["eyesight(right)"] <= 2)
        X["eyesight(right)"] = X["eyesight(right)"] - 1

        X, y = self._filter_rows(X, y, X["systolic"] <= 180)
        X = self._transform_scale(X, "systolic")

        X, y = self._filter_rows(X, y, X["relaxation"] <= 120)
        X = self._transform_scale(X, "relaxation")

        X, y = self._filter_rows(X, y, X["fasting blood sugar"] <= 350)
        X = self._transform_scale(X, "fasting blood sugar")

        X = self._transform_scale(X, "triglyceride")

        X, y = self._filter_rows(X, y, X["HDL"] <= 300)
        X = self._transform_scale(X, "HDL")

        X = self._transform_scale(X, "hemoglobin")

        X, y = self._filter_rows(X, y, X["ALT"] <= 100)
        X = self._transform_scale(X, "ALT")

        X = self._transform_scale(X, "Gtp")

        # Жёстко фиксируем порядок колонок как при обучении.
        X = X[self.feature_columns]

        X_array = X.astype(np.float32).to_numpy()
        y_array = y.astype(np.float32).to_numpy() if y is not None else None
        return X_array, y_array, X.index

    # ------------------------------------------------------------------
    # 3. Разделение данных и tf.data.Dataset
    # ------------------------------------------------------------------
    def prepare_train_validation(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Делит данные на train/validation и выполняет предобработку."""
        data = self._prepare_base_dataframe(df, fit=True)
        target = self.config.target_col

        if target not in data.columns:
            raise ValueError(f"В DataFrame нет целевой колонки '{target}'.")

        X = data.drop(columns=[target])
        y = data[target]

        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        X_train, y_train, _ = self._fit_transform_features(X_train_raw, y_train_raw)
        X_val, y_val, _ = self._transform_features(X_val_raw, y_val_raw)
        return X_train, X_val, y_train, y_val

    def _make_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> tf.data.Dataset:
        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.cast(X, tf.float32), tf.cast(y, tf.float32))
        )
        return dataset.batch(self.config.batch_size)

    # ------------------------------------------------------------------
    # 4. Модель
    # ------------------------------------------------------------------
    def build_model(self, input_dim: int) -> tf.keras.Model:
        """Архитектура из ноутбука."""
        self.model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(128, activation="relu", name="dense_1", kernel_initializer='he_normal'),
                tf.keras.layers.Dropout(0.4, name="dropout_2"),
                tf.keras.layers.Dense(64, activation="relu", name="dense_3", kernel_initializer='he_normal'),
                tf.keras.layers.Dropout(0.3, name="dropout_3"),
                tf.keras.layers.Dense(32, activation="relu", name="dense_4", kernel_initializer='he_normal'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2, name="dropout_4"),
                tf.keras.layers.Dense(16, activation="relu", name="dense_5", kernel_initializer='he_normal'),
                tf.keras.layers.Dropout(0.2, name="dropout_5"),
                tf.keras.layers.Dense(1, activation="sigmoid", name="dense_exit")
            ]
        )
        return self.model

    def compile_model(self):
        if self.model is None:
            raise RuntimeError("Сначала вызови build_model().")

        if self.config.loss_name == "binary_crossentropy":
            loss = tf.keras.losses.BinaryCrossentropy()
        else:
            loss = tf.keras.losses.BinaryFocalCrossentropy()

        self.model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(
                    threshold=self.config.threshold,
                    name="accuracy",
                ),
                tf.keras.metrics.Precision(
                    thresholds=self.config.threshold,
                    name="precision",
                ),
                tf.keras.metrics.Recall(
                    thresholds=self.config.threshold,
                    name="recall",
                ),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
            ],
        )

    # ------------------------------------------------------------------
    # 5. Обучение
    # ------------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        epochs: int = 250,
        use_tensorboard: bool = True,
        use_early_stopping: bool = True,
        verbose: int = 1,
    ):
        """Полный цикл: split -> preprocess -> build -> compile -> train."""
        X_train, X_val, y_train, y_val = self.prepare_train_validation(df)

        train_dataset = self._make_dataset(X_train, y_train)
        val_dataset = self._make_dataset(X_val, y_val)

        self.build_model(input_dim=X_train.shape[1])
        self.compile_model()

        callbacks = []

        if use_tensorboard:
            self.log_dir = "/content/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=self.log_dir,
                    histogram_freq=0,
                    write_graph=False,
                    update_freq="epoch",
                )
            )

        if use_early_stopping:
            callbacks.extend(
                [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_auc",
                        mode="max",
                        patience=25,
                        restore_best_weights=True,
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.5,
                        patience=10,
                        min_lr=1e-6,
                    ),
                ]
            )

        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose,
        )

        return self.history

    # ------------------------------------------------------------------
    # 6. Использование модели
    # ------------------------------------------------------------------
    def preprocess_for_prediction(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.Index]:
        """Подготовка новых данных для predict/predict_proba."""
        data = self._prepare_base_dataframe(df, fit=False)

        # Если случайно передали DataFrame с target, удаляем target.
        if self.config.target_col in data.columns:
            data = data.drop(columns=[self.config.target_col])

        X_array, _, kept_index = self._transform_features(
            data,
            y=None
        )
        return X_array, kept_index

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Возвращает вероятность класса 1."""
        if self.model is None:
            raise RuntimeError("Модель не загружена и не обучена.")

        X, kept_index = self.preprocess_for_prediction(df)
        proba = self.model.predict(X, verbose=0).ravel()
        return pd.Series(proba, index=kept_index, name="smoking_probability")

    def predict(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> pd.DataFrame:
        """Возвращает вероятность и итоговый класс 0/1."""
        threshold = self.config.threshold if threshold is None else threshold
        proba = self.predict_proba(df)
        pred = (proba >= threshold).astype(int)
        return pd.DataFrame(
            {
                "smoking_probability": proba,
                "smoking_prediction": pred,
            }
        )
    # ------------------------------------------------------------------
    # 7. Сохранение и загрузка
    # ------------------------------------------------------------------
    def save(self, folder: str):
        """Сохраняет всю связку: модель + scalers + конфиг + порядок колонок."""
        if self.model is None:
            raise RuntimeError("Нечего сохранять: модель ещё не обучена.")

        os.makedirs(folder, exist_ok=True)

        self.model.save(os.path.join(folder, "model.keras"))

        state = {
            "config": asdict(self.config),
            "scalers": self.scalers,
            "feature_columns": self.feature_columns,
        }
        joblib.dump(state, os.path.join(folder, "preprocessing.joblib"))

        with open(os.path.join(folder, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, folder: str) -> "SmokingNNPipeline":
        """Загружает pipeline после save()."""
        state = joblib.load(os.path.join(folder, "preprocessing.joblib"))
        config = SmokingPipelineConfig(**state["config"])

        pipe = cls(config=config)
        pipe.scalers = state["scalers"]
        pipe.feature_columns = state["feature_columns"]
        pipe.model = tf.keras.models.load_model(os.path.join(folder, "model.keras"), compile=False)
        pipe.compile_model()
        return pipe

    def save_weights(self, folder: str):
        """Сохраняет только веса сети + параметры предобработки."""
        if self.model is None:
            raise RuntimeError("Нечего сохранять: модель ещё не обучена.")

        os.makedirs(folder, exist_ok=True)
        self.model.save_weights(os.path.join(folder, "weights.weights.h5"))

        state = {
            "config": asdict(self.config),
            "scalers": self.scalers,
            "feature_columns": self.feature_columns,
            "input_dim": len(self.feature_columns),
        }
        joblib.dump(state, os.path.join(folder, "preprocessing_and_arch.joblib"))

    @classmethod
    def load_weights(cls, folder: str) -> "SmokingNNPipeline":
        """Загружает веса сети, предварительно восстановив архитектуру."""
        state = joblib.load(os.path.join(folder, "preprocessing_and_arch.joblib"))
        config = SmokingPipelineConfig(**state["config"])

        pipe = cls(config=config)
        pipe.scalers = state["scalers"]
        pipe.feature_columns = state["feature_columns"]
        pipe.build_model(input_dim=state["input_dim"])
        pipe.compile_model()
        pipe.model.load_weights(os.path.join(folder, "weights.weights.h5"))
        return pipe


# ----------------------------------------------------------------------
# Пример использования:
# ----------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("smoking.csv")

    config = SmokingPipelineConfig(
        threshold=0.51,
        loss_name="binary_focal_crossentropy",
        batch_size=128,
        test_size=0.30,
        random_state=9,
    )

    pipe = SmokingNNPipeline(config)
    pipe.fit(df, epochs=250)

    # Сохранение всей модели и предобработки
    pipe.save("smoking_nn_pipeline")

    # Загрузка
    loaded_pipe = SmokingNNPipeline.load("smoking_nn_pipeline")

    # Использование
    predictions = loaded_pipe.predict(df.head(5))
    print(predictions)
