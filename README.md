# Smoking Prediction

Сервис, определяющий по биомедицинским показателям пациента (антропометрия,
давление, липидный профиль, печёночные ферменты и т.д.), курит ли человек.
Полный цикл: исследование и предобработка данных → обучение нейронной сети →
REST API → контейнеризация.

## Стек

`Python` · `pandas` / `NumPy` · `scikit-learn` (RobustScaler, StandardScaler) ·
`TensorFlow / Keras` · `FastAPI` · `Docker`

## Задача

Бинарная классификация: по 16 числовым, категориальным и бинарным признакам
предсказать `smoking ∈ {0, 1}`. Датасет — [Body Signal of Smoking](https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking)
(Kaggle, 55 692 записи, результаты медицинских осмотров). Подробности —
[docs/dataset.md](docs/dataset.md).

## Подход

- **Предобработка** — удаление неинформативных признаков по корреляции с
  целевой переменной, отсечение выбросов на основе клинических референсных
  диапазонов (артериальное давление, глюкоза, ЛПВП, АЛТ, острота зрения),
  масштабирование (`RobustScaler` / `StandardScaler`, с логарифмированием
  скошенных распределений).
- **Модель** — полносвязная нейронная сеть (MLP, 4 скрытых слоя) с Dropout и
  BatchNormalization, обучена с фокальной кросс-энтропией (Focal Loss) для
  компенсации дисбаланса классов (1,72 : 1). Подробности — [docs/model.md](docs/model.md).
- **API** — FastAPI-приложение с формой ввода (`GET /`) и эндпойнтом
  предсказания (`POST /send-data`), двухуровневая валидация входных данных
  (Pydantic + защитная проверка пайплайна). Подробности — [docs/api.md](docs/api.md).
- **Развёртывание** — контейнеризовано в Docker (`python:3.11-slim`,
  health-check, однопроцессный запуск `uvicorn` под TensorFlow).

## Структура репозитория

```
.
├── main.py                    # FastAPI-приложение (эндпойнты GET /, POST /send-data)
├── smoking_pipeline.py         # SmokingNNPipeline: предобработка, обучение, инференс, save/load
├── templates/index.html        # Веб-форма ввода показателей
├── smoking_nn_pipeline/        # Артефакты обученной модели (веса, конфиг, препроцессинг)
├── notebooks/                  # Исследовательский анализ данных и обучение модели
├── docs/                       # Документация: датасет, модель, API
├── smoking.csv                 # Исходный датасет
├── Dockerfile / .dockerignore
└── requirements.txt
```

## Запуск

### Локально

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Открыть `http://localhost:8000/`.

### Docker

```bash
docker build -t smoking-predictor .
docker run -p 8000:8000 smoking-predictor
```

## Документация

- [docs/dataset.md](docs/dataset.md) — датасет, предобработка, обоснование отсечения выбросов
- [docs/model.md](docs/model.md) — архитектура сети, функция потерь, метрики, гиперпараметры
- [docs/api.md](docs/api.md) — спецификация API, примеры запросов, тестирование

## Источники

1. Badr I., Hamadouche M. A., Bouali F. et al. Explainable artificial intelligence driven insights into smoking prediction using machine learning and clinical parameters // Scientific Reports. — 2025. — Vol. 15, № 1. — Art. 22285. — DOI: 10.1038/s41598-025-09409-w
2. Lin T.-Y., Goyal P., Girshick R., He K., Dollár P. Focal Loss for Dense Object Detection // Proceedings of the IEEE International Conference on Computer Vision (ICCV). — 2017. — P. 2980–2988. — DOI: 10.1109/ICCV.2017.324
3. Кобалава Ж. Д., Конради А. О., Недогода С. В. и др. Артериальная гипертензия у взрослых. Клинические рекомендации 2020 // Российский кардиологический журнал. — 2020. — Т. 25, № 3. — С. 149–218. — DOI: 10.15829/1560-4071-2020-3-3786
4. Оганов Р. Г. Влияние курения на здоровье населения: место России в Европе // Профилактика заболеваний и укрепление здоровья. — 2002. — № 6. — С. 17–20
5. Камышников В. С. Клинические лабораторные тесты от А до Я и их диагностические профили: справочное пособие. — М.: МЕДпресс-информ, 2009. — 320 с.
6. Kukuroo3. Body Signal of Smoking [Электронный ресурс] // Kaggle. — 2022. — URL: https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking
7. Mousavi M., Maghsoudi M., Dehghani A. et al. Association between lipid profiles and cigarette smoke among adults in the Persian cohort (Shahedieh) study // BMC Public Health. — 2024. — Vol. 24, № 1. — Art. 1236. — DOI: 10.1186/s12889-024-18734-0
