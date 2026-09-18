"""
Smoking Prediction API
FastAPI-приложение с двумя эндпойнтами:
  GET  /           — HTML-страница с формой (templates/index.html)
  POST /send-data  — принимает JSON, возвращает предсказание (курит / не курит)
                     или 400/422, если данные нарушают пороги обучающей выборки.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from smoking_pipeline import SmokingNNPipeline

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
BASE_DIR     = Path(__file__).parent
MODEL_FOLDER = BASE_DIR / "smoking_nn_pipeline"
INDEX_HTML   = BASE_DIR / "templates" / "index.html"

pipeline: SmokingNNPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружаем пайплайн один раз при старте приложения."""
    global pipeline
    pipeline = SmokingNNPipeline.load(str(MODEL_FOLDER))
    yield


app = FastAPI(
    title="Smoking Prediction API",
    description="Определяет, курит ли человек, по биомедицинским показателям.",
    version="1.0.0",
    lifespan=lifespan,
    debug=True
)

# ---------------------------------------------------------------------------
# Pydantic-схема входных данных
# Ограничения le/ge взяты напрямую из правил фильтрации обучающей выборки:
# строки, нарушающие их, были отброшены при обучении → модель не умеет
# с ними работать → 422 ещё до вызова пайплайна.
# ---------------------------------------------------------------------------

class SmokingInput(BaseModel):
    gender:              Literal["M", "F"]
    age:                 float = Field(gt=0)
    height_cm:           float = Field(gt=0)
    weight_kg:           float = Field(gt=0)
    waist_cm:            float = Field(gt=0)
    eyesight_left:       float = Field(ge=0, le=2)
    eyesight_right:      float = Field(ge=0, le=2)
    systolic:            float = Field(gt=0, le=180)
    relaxation:          float = Field(gt=0, le=120)
    fasting_blood_sugar: float = Field(gt=0, le=350)
    triglyceride:        float = Field(gt=0)
    HDL:                 float = Field(gt=0, le=300)
    hemoglobin:          float = Field(gt=0)
    ALT:                 float = Field(gt=0, le=100)
    Gtp:                 float = Field(gt=0)
    tartar:              Literal["Y", "N"]
    dental_caries:       Literal["Y", "N"]


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def to_dataframe(data: SmokingInput) -> pd.DataFrame:
    """Перекладывает Pydantic-объект в DataFrame с именами колонок пайплайна."""
    return pd.DataFrame([{
        "gender":              data.gender,
        "age":                 data.age,
        "height(cm)":          data.height_cm,
        "weight(kg)":          data.weight_kg,
        "waist(cm)":           data.waist_cm,
        "eyesight(left)":      data.eyesight_left,
        "eyesight(right)":     data.eyesight_right,
        "systolic":            data.systolic,
        "relaxation":          data.relaxation,
        "fasting blood sugar": data.fasting_blood_sugar,
        "triglyceride":        data.triglyceride,
        "HDL":                 data.HDL,
        "hemoglobin":          data.hemoglobin,
        "ALT":                 data.ALT,
        "Gtp":                 data.Gtp,
        "tartar":              data.tartar,
        "dental caries":       data.dental_caries,
    }])


_html_cache: str | None = None

def _read_html() -> str:
    """Читает HTML-файл один раз и кэширует в памяти."""
    global _html_cache
    if _html_cache is None:
        _html_cache = INDEX_HTML.read_text(encoding="utf-8")
    return _html_cache


# ---------------------------------------------------------------------------
# Эндпойнты
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, summary="Форма ввода данных")
async def index() -> HTMLResponse:
    """Возвращает HTML-страницу с формой (templates/index.html)."""
    return HTMLResponse(content=_read_html())


@app.post(
    "/send-data",
    summary="Предсказание курения",
    responses={
        200: {
            "description": "Успешное предсказание",
            "content": {
                "application/json": {
                    "example": {
                        "smoking": True,
                        "probability": 0.8123,
                        "label": "Курит",
                    }
                }
            },
        },
        400: {
            "description": "Данные отфильтрованы пайплайном (edge case)",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "errors": ["Запись была отфильтрована во время предобработки."]
                        }
                    }
                }
            },
        },
        422: {
            "description": "Некорректные данные (нарушены пороги Pydantic)",
        },
    },
)
async def send_data(data: SmokingInput):
    """
    Принимает JSON с биомедицинскими показателями и возвращает предсказание.

    Валидация происходит в два слоя:
    - **422** — Pydantic отклоняет поля вне допустимых диапазонов.
    - **400** — пайплайн вернул пустой DataFrame (edge case, если строка
      была отфильтрована уже внутри `_transform_features`).
    """
    df = to_dataframe(data)
    #try:
    result = pipeline.predict(df)
    #except Exception as exc:
    #    raise HTTPException(status_code=500, detail=str(exc)) from exc

    if result.empty:
        raise HTTPException(
            status_code=400,
            detail={
                "errors": [
                    "Запись была отфильтрована во время предобработки. "
                    "Проверьте, что все значения находятся в допустимых диапазонах."
                ]
            },
        )

    row      = result.iloc[0]
    smoking  = bool(row["smoking_prediction"])
    prob     = round(float(row["smoking_probability"]), 4)

    return {
        "smoking":     smoking,
        "probability": prob,
        "label":       "Курит" if smoking else "Не курит",
    }
