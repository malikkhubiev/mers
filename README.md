<div align="center">

<img src="https://malikkhubiev.github.io/benz/merscedes.png" width="900" alt="Mercedes-Benz Price Intelligence" />

# Mercedes-Benz Price Intelligence

**From raw Avito listings to a multi-model ML experiment with MLflow and PyTorch.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-experiments-0194E2?logo=mlflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-regression-EE4C2C?logo=pytorch&logoColor=white)
![Sklearn](https://img.shields.io/badge/scikit--learn-baselines-F7931E?logo=scikitlearn&logoColor=white)

</div>

---

## Table of Contents (EN)

1. [Project Overview](#project-overview)
2. [Business Motivation](#business-motivation)
3. [Data Pipeline](#data-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Modeling and Experiments](#modeling-and-experiments)
6. [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
7. [PyTorch Architecture](#pytorch-architecture)
8. [Evaluation, Charts and Diagnostics](#evaluation-charts-and-diagnostics)
9. [Saved Artifacts for Streamlit](#saved-artifacts-for-streamlit)
10. [Repository Structure](#repository-structure)

---

## Project Overview

This repository contains a complete experimental pipeline for **Mercedes-Benz used car price prediction** based on real Avito listings.  
The core notebook converted to script is `benz_pytorch___mlflow___sklearn.py`, which implements:

- Full data cleaning and parsing from a pre-scraped `clean_mers.csv`.
- Advanced feature engineering (log-transforms, interaction terms, price segments, popularity indicators).
- A **multi-model benchmark** of classic ML algorithms (Linear/Ridge/Lasso, Random Forest, Gradient Boosting, XGBoost, CatBoost).
- A custom **PyTorch residual network** with LayerNorm, dropout and learning-rate scheduling.
- Unified **experiment tracking in MLflow** with metrics in rubles (R², MAE, RMSE, MAPE).
- Export of best model, scaler and metadata into `saved_models/` for integration with a Streamlit application.

The focus is not only on achieving a strong metric, but on building a **reproducible, auditable and extendable experiment stack**.

---

## Business Motivation

- **Dealers and brokers** need a realistic price reference for Mercedes-Benz models across Russian regions instead of relying on noisy listing prices.
- **Analysts and data scientists** need a transparent benchmark: which algorithms and architectures perform best on this particular domain.
- **Product and ML engineers** need a robust training and evaluation pipeline that can later be wired into production services (API or Streamlit UI).

The pipeline is designed so that:

- New Avito data snapshots can be plugged in with minimal changes.
- Models and experiments are **versioned in MLflow**.
- Best models are exported in a format directly consumable by a serving layer.

---

## Data Pipeline

Source data is assumed to be stored in `clean_mers.csv`, obtained from a custom web scraper over Avito listings.

Key transformation steps in `benz_pytorch___mlflow___sklearn.py`:

- **Price cleaning**  
  Function `clean_price` removes currency symbols and all whitespace types (including non-breaking spaces), converts prices to integers, and filters out obviously invalid values:
  - Keep only prices in the range `[100_000, 40_000_000]` RUB.

- **Parsing vehicle description**  
  Function `parse_full_name` parses the string like  
  `"Mercedes-Benz C-класс 1.6 AT, 2019, 97 000 км"` into:
  - `model` (e.g. `Mercedes-Benz C-класс`),
  - `year` (int),
  - `mileage` (int, km).

- **Basic sanity checks and filters**
  - Remove rows with unreasonable `year` (`year <= 1900`).
  - Drop rows with non-positive prices.

- **Age computation**
  - `age = current_year - year`, with `current_year = 2026` in the script.

The resulting cleaned dataset `df` has columns like:

| Column      | Type    | Description                                  |
|------------|---------|----------------------------------------------|
| full_name  | string  | Raw listing text from Avito                  |
| price      | int     | Cleaned price in RUB                         |
| details    | string  | Additional listing details                   |
| rating     | string  | Textual rating or quality indicator          |
| model      | string  | Parsed model name                            |
| year       | int     | Year of manufacture                          |
| mileage    | int     | Mileage in km                                |
| age        | int     | Vehicle age (years)                          |

---

## Feature Engineering

All advanced features are created in `df_feat` and are later used for both classical ML and PyTorch models.

Key feature blocks:

1. **Model-level statistics**

   Grouping by `model`:

   - `model_price_mean`, `model_price_median`, `model_price_std`, `model_count`.
   - These act as **dynamic baselines** and encode the typical price and variability per model.

2. **Log-transforms**

   To stabilize skewed distributions:

   - `log_price = log1p(price)` (used as the main target),
   - `log_mileage = log1p(mileage)`,
   - `log_model_price_mean = log1p(model_price_mean)`.

3. **Interaction and ratio features**

   - `mileage_per_year = mileage / (age + 1)`,
   - `price_per_year = price / (age + 1)`,
   - `mileage_x_age = mileage * age`.

4. **Binary flags**

   - `is_new` (age ≤ 3),
   - `is_almost_new` (3 < age ≤ 7),
   - `high_mileage` (mileage > 150_000),
   - `very_high_mileage` (mileage > 300_000),
   - `is_premium` (model above median model price).

5. **Popularity and segments**

   - `model_popularity = model_count / len(df_feat)`,
   - Price segments with `pd.cut` into:
     `Economy`, `Budget`, `Mid`, `Premium`, `Luxury`, `Ultra-Luxury`,
   - One-hot encoding for price segments and decades (`decade_*`, `segment_*`).

The final feature list (`existing_features`) includes:

- Core numeric features (`year`, `age`, `year_rank`),
- Model statistics,
- Log-transformed and interaction features,
- Binary indicators,
- One-hot encoded `decade_*` and `segment_*` features.

These features are then:

- Split into train/test via `train_test_split`.
- Scaled with `RobustScaler` (robust to outliers), both for classical models and as input to PyTorch.

---

## Modeling and Experiments

The script defines a **multi-model benchmark** with a consistent evaluation interface.

### Target and Metrics

- **Target**: `log_price = log1p(price)`.  
  Predictions are converted back to RUB via `expm1` for business metrics.

- **Business-level metrics** (computed in `evaluate_model_rub`):
  - R² on real prices,
  - MAE (RUB),
  - RMSE (RUB),
  - MAPE (%).

### Classical ML models (scikit-learn, XGBoost, CatBoost)

Each configuration is logged into MLflow under a separate run:

| Name             | Library        | Notes                                      |
|------------------|----------------|--------------------------------------------|
| LinearRegression | scikit-learn   | Simple linear baseline                     |
| Ridge            | scikit-learn   | L2-regularized regression                  |
| Lasso            | scikit-learn   | Sparse linear model                        |
| RandomForest     | scikit-learn   | Non-linear ensemble, no scaling required   |
| GradientBoosting | scikit-learn   | Gradient boosted trees                     |
| XGBoost          | xgboost        | Strong tree ensemble baseline              |
| CatBoost         | catboost       | Gradient boosting with advanced handling   |

For each model:

- Data is chosen either in scaled or raw form (`use_scaled` flag).
- The model is trained on the train split.
- Train and test metrics in RUB-space are computed and logged.
- The trained model is stored as an MLflow artifact.

All test metrics are collected into a comparison table:

- `comparison_df = pd.DataFrame(all_results).T`,  
- Saved to `saved_models/model_comparison_<timestamp>.csv`.

---

## Experiment Tracking with MLflow

The experiment name is set to `"Mercedes-Benz Price Prediction"`, with local artifact storage in `./mlruns`.

For each model:

- Parameters (`alpha`, `n_estimators`, `max_depth`, `learning_rate`, etc.) are logged.
- Metrics are logged with consistent names across runs:
  - `train_r2`, `train_mae`, `train_rmse`, `train_mape`,
  - `test_r2`, `test_mae`, `test_rmse`, `test_mape`.
- The model object is logged using `mlflow.sklearn.log_model` (for sklearn-based models).

You can inspect all experiments with:

```bash
mlflow ui --backend-store-uri ./mlruns
```

and open the MLflow UI in your browser to compare runs, hyperparameters and metrics.

---

## PyTorch Architecture

The core neural network model is `ImprovedPricePredictor`, defined as:

- Input: `input_dim = len(existing_features)` features (after scaling).
- Several fully connected hidden layers with dimensions `[256, 128, 64, 32]` by default.
- Layer normalization (`LayerNorm`) and dropout after each linear layer.
- Residual connections between hidden layers (with identity or linear projection when dimensions change).
- Xavier/Glorot initialization for all linear layers.
- Final linear output layer with a single neuron predicting `log_price`.

Training is handled by `train_pytorch_model`:

- Optimizer: `AdamW` with weight decay.
- Learning rate scheduler: `ReduceLROnPlateau` on validation loss.
- Early stopping on **R² on test set** with patience.
- Gradient clipping to stabilize training.
- Detailed tracking of:
  - train/test losses,
  - train/test R² per epoch.

The script prints:

- Model architecture summary (input size, hidden layers, parameter count),
- Best achieved R² for the PyTorch model on the held-out test set.

---

## Evaluation, Charts and Diagnostics

Several visualization blocks are produced in the script to analyze both **model comparison** and **error structure**.

### Model comparison charts

Based on `comparison_df`:

- **R² comparison**  
  Horizontal bar chart of R² scores for all models.

- **MAE comparison**  
  Horizontal bar chart of MAE in RUB.

- **MAPE comparison**  
  Horizontal bar chart of MAPE (%).

These plots clearly show which algorithms dominate on this dataset (often boosting or the residual PyTorch model).

### PyTorch training curves

A 2x2 figure shows:

- Train vs test loss curves (log scale) per epoch.
- Train vs test R² per epoch, with the XGBoost R² added as a horizontal reference line when available.
- Predicted vs actual prices scatter plot with identity line.
- Histogram of residuals with mean and ±1σ bands.

These diagnostics make it easy to:

- Detect overfitting / underfitting,
- Verify stability of the training process,
- Check how well the model respects the identity line on prices.

### Residuals analysis for the best model

After selecting the best model from `comparison_df` (sklearn or PyTorch), the script:

- Computes residuals in RUB and in percent for the **entire dataset**.
- Prints summary statistics:
  - Mean, median, standard deviation,
  - Distribution of residuals within 1σ and 2σ,
  - Absolute and relative extremes.
- Builds a 2x3 diagnostic figure:
  - Predicted vs actual,
  - Residuals histogram with mean and ±σ,
  - Q-Q plot against the normal distribution,
  - Residuals vs predictions (check for heteroscedasticity),
  - Relative residuals vs predictions,
  - Boxplots of residuals per price segment (`<1M`, `1-2M`, `2-3M`, `3-5M`, `5-10M`, `>10M`).

This allows to see **where** the model tends to over- or under-price certain segments and whether the error distribution is acceptable for business use.

---

## Saved Artifacts for Streamlit

Once the best model is identified, the script saves a complete set of artifacts to `saved_models/`:

- **Best model**:
  - PyTorch: `.pth` with `state_dict` plus a separate JSON-like description of architecture.
  - Tree-based models: `.joblib` with `feature_importances_` when available.
  - Linear models: `.pkl` with coefficients mapped to feature names.
- **Scaler**: `scaler_*.pkl` (RobustScaler fitted on training data).
- **Model statistics**: `model_stats_*.pkl` with per-model price statistics.
- **Config metadata**: `model_config_*.json` with:
  - model name, training date, metrics, feature list, data size,
  - type of artifact (`pytorch`, `joblib`, `pickle`), feature importance or coefficients when present.
- **Full pipeline object**: `full_pipeline_*.pkl` with the information needed to reproduce the prediction pipeline.
- **Streamlit-friendly pipeline**: `streamlit_model_*.pkl` with:
  - model path, scaler path, feature names, user-friendly feature labels.
- **Model comparison CSV**: `model_comparison_*.csv` for historical tracking of experiments.

These artifacts are designed to be consumed by a Streamlit application that:

- Accepts model, year and mileage as input,
- Prepares the feature vector using the exact same feature list and scaler,
- Loads the best model and outputs a price prediction in RUB.

---

## Repository Structure

Suggested structure for this project:

```text
.
├── benz_pytorch___mlflow___sklearn.py   # Main experiment script (Colab notebook exported to .py)
├── clean_mers.csv                       # Preprocessed Avito data snapshot (not included in repo by default)
├── saved_models/                        # Exported models, scalers, metadata, pipelines
├── mlruns/                              # Local MLflow experiment store
├── merscedes_app.py                     # Streamlit serving application (separate UI layer)
├── requirements.txt                     # Python dependencies
└── README.md                            # This documentation
```

Depending on your environment, `clean_mers.csv` and some artifact folders may be generated locally and are not necessarily tracked in Git.

---

## Оглавление (RU)

1. [Общая идея проекта](#общая-идея-проекта)
2. [Бизнес-мотивация](#бизнес-мотивация)
3. [Конвейер данных](#конвейер-данных)
4. [Инженерия признаков](#инженерия-признаков)
5. [Модели и эксперименты](#модели-и-эксперименты)
6. [Отслеживание экспериментов в MLflow](#отслеживание-экспериментов-в-mlflow)
7. [Архитектура PyTorch-модели](#архитектура-pytorch-модели)
8. [Оценка, графики и диагностика](#оценка-графики-и-диагностика)
9. [Сохранённые артефакты для Streamlit](#сохранённые-артефакты-для-streamlit)
10. [Структура репозитория](#структура-репозитория)

---

## Общая идея проекта

Этот репозиторий содержит полный экспериментальный пайплайн для **прогнозирования стоимости подержанных автомобилей Mercedes-Benz** по реальным объявлениям Avito.  
Основной код находится в файле `benz_pytorch___mlflow___sklearn.py` и реализует:

- Полную очистку и парсинг данных из предварительно подготовленного `clean_mers.csv`.
- Продвинутую инженерию признаков (логарифмы, взаимодействия, ценовые сегменты, популярность моделей).
- Мульти-модельный бенчмарк классических алгоритмов (Linear/Ridge/Lasso, RandomForest, GradientBoosting, XGBoost, CatBoost).
- Кастомную **нейросеть на PyTorch** с остаточными связями, LayerNorm, dropout и адаптивной скоростью обучения.
- Единое **отслеживание экспериментов в MLflow** с бизнес-метриками в рублях (R², MAE, RMSE, MAPE).
- Экспорт лучшей модели, скейлера и метаданных в `saved_models/` для последующей интеграции со Streamlit-приложением.

Фокус не только на метрике, но и на **воспроизводимости, прозрачности и готовности к продакшену**.

---

## Бизнес-мотивация

- **Дилеры и брокеры** получают реалистичный ориентир по ценам на Mercedes-Benz в разных регионах РФ, а не «шум» из объявлений.
- **Аналитики и дата-сайентисты** получают понятный стенд: какие алгоритмы и архитектуры лучше работают именно на этом домене.
- **Инженеры и ML-разработчики** получают устойчивый тренировочный и оценочный пайплайн, который можно обернуть в API или UI.

Пайплайн собран так, чтобы:

- Легко подключать новые срезы данных Avito.
- **Версионировать эксперименты в MLflow**.
- Выгружать лучшие модели в формат, готовый к использованию в сервисе.

---

## Конвейер данных

Исходные данные ожидаются в файле `clean_mers.csv`, полученном из веб-скрапера Avito.

Ключевые этапы обработки в `benz_pytorch___mlflow___sklearn.py`:

- **Очистка цены**
  - Функция `clean_price` убирает символы валюты и любые пробелы, корректно обрабатывает неразрывные пробелы.
  - Значения конвертируются в `int`, фильтруются аномально низкие и высокие цены (`100 000` – `40 000 000` руб).

- **Парсинг описания автомобиля**
  - Функция `parse_full_name` разбирает строку формата  
    `"Mercedes-Benz C-класс 1.6 AT, 2019, 97 000 км"` на:
    - `model` (пример: `Mercedes-Benz C-класс`),
    - `year`,
    - `mileage` (км).

- **Фильтрация и sanity-check**
  - Удаляются строки с нереалистичным годом (`year <= 1900`).
  - Удаляются строки с нулевой или отрицательной ценой.

- **Возраст автомобиля**
  - `age = current_year - year`, где `current_year = 2026` (параметр можно изменить при переносе во времени).

В итоге получается очищенный датафрейм `df` со столбцами:

| Колонка   | Тип    | Описание                                  |
|----------|--------|-------------------------------------------|
| full_name| string | Исходный текст из объявления Avito        |
| price    | int    | Очищенная цена в рублях                   |
| details  | string | Дополнительные детали объявления          |
| rating   | string | Текстовый рейтинг или маркер качества     |
| model    | string | Распарсенная модель                       |
| year     | int    | Год выпуска                               |
| mileage  | int    | Пробег в км                               |
| age      | int    | Возраст автомобиля                        |

---

## Инженерия признаков

Все продвинутые признаки собираются в `df_feat` и далее используются как для классических моделей, так и для PyTorch.

Основные блоки признаков:

1. **Статистика по моделям**

   Группировка по `model`:

   - `model_price_mean`, `model_price_median`, `model_price_std`, `model_count`.
   - Это динамические «базовые» цены и дисперсия по каждой модели.

2. **Логарифмические преобразования**

   Для борьбы со скошенными распределениями:

   - `log_price = log1p(price)` (основная целевая переменная),
   - `log_mileage = log1p(mileage)`,
   - `log_model_price_mean = log1p(model_price_mean)`.

3. **Взаимодействия и отношения**

   - `mileage_per_year = mileage / (age + 1)`,
   - `price_per_year = price / (age + 1)`,
   - `mileage_x_age = mileage * age`.

4. **Бинарные признаки**

   - `is_new` (возраст ≤ 3 лет),
   - `is_almost_new` (3 < возраст ≤ 7 лет),
   - `high_mileage` (пробег > 150 000 км),
   - `very_high_mileage` (пробег > 300 000 км),
   - `is_premium` (модель дороже медианы по модельной цене).

5. **Популярность и сегменты**

   - `model_popularity = model_count / len(df_feat)`,
   - Ценовые сегменты через `pd.cut`:  
     `Economy`, `Budget`, `Mid`, `Premium`, `Luxury`, `Ultra-Luxury`,
   - One-hot кодирование ценовых сегментов и декад (`decade_*`, `segment_*`).

Финальный список признаков (`existing_features`) содержит:

- Базовые числовые признаки (`year`, `age`, `year_rank`),
- Статистики по моделям,
- Логарифмы и взаимодействия,
- Бинарные индикаторы,
- One-hot признаки по сегментам и декадам.

Далее:

- Данные делятся на train/test (`train_test_split`).
- Признаки масштабируются с помощью `RobustScaler` (устойчив к выбросам) и используются и в классических моделях, и в PyTorch.

---

## Модели и эксперименты

Скрипт реализует **мульти-модельный бенчмарк** через единый интерфейс.

### Таргет и метрики

- **Таргет**: `log_price = log1p(price)`.  
  При расчёте бизнес-метрик предсказания переводятся обратно в рубли (`expm1`).

- **Бизнес-метрики** (функция `evaluate_model_rub`):
  - R² по реальным ценам,
  - MAE (руб),
  - RMSE (руб),
  - MAPE (%).

### Классические модели (scikit-learn, XGBoost, CatBoost)

Каждая конфигурация логируется в MLflow отдельным запуском:

| Имя             | Библиотека      | Особенности                               |
|-----------------|-----------------|-------------------------------------------|
| LinearRegression| scikit-learn    | Базовая линейная регрессия               |
| Ridge           | scikit-learn    | L2-регуляризация                          |
| Lasso           | scikit-learn    | Разряженная линейная модель               |
| RandomForest    | scikit-learn    | Нелинейный ансамбль деревьев              |
| GradientBoosting| scikit-learn    | Градиентный бустинг                       |
| XGBoost         | xgboost         | Сильный бустинг-бейзлайн                  |
| CatBoost        | catboost        | Бустинг с продвинутой обработкой признаков|

Для каждой модели:

- Выбирается масштабированная или исходная матрица признаков (`use_scaled`).
- Модель обучается на train-части.
- Считаются метрики в рублях для train и test.
- Объект модели сохраняется как артефакт MLflow.

Тестовые метрики собираются в таблицу:

- `comparison_df = pd.DataFrame(all_results).T`,
- Сохраняется в `saved_models/model_comparison_<timestamp>.csv`.

---

## Отслеживание экспериментов в MLflow

Эксперимент называется `"Mercedes-Benz Price Prediction"`, артефакты сохраняются в `./mlruns`.

Для каждого запуска:

- Логируются параметры (`alpha`, `n_estimators`, `max_depth`, `learning_rate` и др.).
- Логируются метрики:
  - `train_r2`, `train_mae`, `train_rmse`, `train_mape`,
  - `test_r2`, `test_mae`, `test_rmse`, `test_mape`.
- Модель сохраняется через `mlflow.sklearn.log_model` (для sklearn-моделей).

Интерфейс MLflow UI позволяет:

- Сравнивать модели по метрикам,
- Смотреть артефакты (сохранённые модели, графики),
- Фильтровать/сортировать эксперименты по гиперпараметрам.

Запуск UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

---

## Архитектура PyTorch-модели

Основная нейросеть `ImprovedPricePredictor`:

- Вход: `input_dim = len(existing_features)` отмасштабированных признаков.
- Скрытые слои: по умолчанию `[256, 128, 64, 32]`.
- После каждого полносвязного слоя:
  - `LayerNorm`,
  - `ReLU`,
  - `Dropout`.
- Остаточные связи между слоями (identity или линейный слой для выравнивания размерностей).
- Инициализация весов по Xavier/Glorot.
- Выход: один нейрон, прогнозирующий `log_price`.

Функция обучения `train_pytorch_model`:

- Оптимизатор `AdamW` с weight decay.
- Планировщик скорости обучения `ReduceLROnPlateau` по валидационной потере.
- Ранний останов по R² на валидации (test) с параметром patience.
- Градиентный клиппинг для устойчивости.
- Логируются:
  - train/test loss по эпохам,
  - train/test R² по эпохам.

Скрипт печатает:

- Краткое описание архитектуры (размер входа, слои, количество параметров),
- Лучшую достигнутую метрику R² на отложенной выборке.

---

## Оценка, графики и диагностика

В скрипте реализован набор визуализаций, позволяющий оценить качество моделей.

### Диаграммы сравнения моделей

На основе `comparison_df` строятся:

- Гистограмма R² по моделям (горизонтальные бары),
- Гистограмма MAE в рублях,
- Гистограмма MAPE в процентах.

Эти графики позволяют быстро понять, какие алгоритмы лучше работают на данных Avito.

### История обучения PyTorch

В 2x2 сетке отображаются:

- Кривые train/test loss (в логарифмическом масштабе),
- Кривые train/test R², плюс горизонтальная линия с R² лучшего бустинга (например, XGBoost),
- Диаграмма рассеяния «факт против прогноза» с линией идеального соответствия,
- Гистограмма остатков с линиями среднего и ±1σ.

Это помогает:

- Диагностировать переобучение/недообучение,
- Проверить стабильность тренировки,
- Оценить, насколько плотно точки лежат вдоль диагонали.

### Анализ остатков лучшей модели

После выбора лучшей модели (из `comparison_df`) рассчитываются:

- Остатки в рублях и в процентах для всех наблюдений,
- Сводная статистика:
  - Среднее, медиана, стандартное отклонение,
  - Доля наблюдений в пределах 1σ и 2σ,
  - Минимальная/максимальная ошибка и относительные значения.

Строится 2x3 набор графиков:

- Предсказание против факта,
- Гистограмма остатков с линиями среднего и ±σ,
- Q-Q plot остатков,
- Остатки против предсказаний (проверка гетероскедастичности),
- Относительные остатки против предсказаний,
- Boxplot остатков по ценовым сегментам (`<1M`, `1-2M`, `2-3M`, `3-5M`, `5-10M`, `>10M`).

Отдельно выводится анализ выбросов (по z-score) с примерами конкретных автомобилей, где ошибка особенно велика.

---

## Сохранённые артефакты для Streamlit

После выбора лучшей модели скрипт сохраняет полный набор артефактов в `saved_models/`:

- **Лучшая модель**:
  - Для PyTorch: `.pth` с `state_dict` и отдельным описанием архитектуры.
  - Для деревьев/ансамблей: `.joblib` с важностями признаков (если есть).
  - Для линейных моделей: `.pkl` с коэффициентами по признакам.
- **Скейлер**: `scaler_*.pkl` (RobustScaler, натренированный на обучающей выборке).
- **Статистика по моделям**: `model_stats_*.pkl` с агрегатами по `model`.
- **Конфигурация и метаданные**: `model_config_*.json`:
  - имя модели, дата обучения, метрики,
  - список признаков и размер выборки,
  - тип артефакта (`pytorch`, `joblib`, `pickle`),
  - важности признаков или коэффициенты, если доступны.
- **Полный пайплайн**: `full_pipeline_*.pkl` — всё, что нужно, чтобы повторить инференс.
- **Упрощённый пайплайн для Streamlit**: `streamlit_model_*.pkl`:
  - пути к модели и скейлеру,
  - список признаков,
  - читабельные имена признаков для отображения в UI.
- **Сравнение моделей**: `model_comparison_*.csv` для истории экспериментов.

Эти артефакты можно напрямую загрузить в Streamlit-приложение, которое:

- Принимает модель, год и пробег,
- Формирует вектор признаков в точном соответствии с `existing_features`,
- Применяет тот же скейлер и лучшую модель,
- Возвращает прогноз цены в рублях.

---

## Структура репозитория

Рекомендуемая структура:

```text
.
├── benz_pytorch___mlflow___sklearn.py   # Основной экспериментальный скрипт
├── clean_mers.csv                       # Снимок данных Avito (локальный артефакт)
├── saved_models/                        # Лучшие модели, скейлеры, пайплайны, метаданные
├── mlruns/                              # Локальное хранилище MLflow
├── merscedes_app.py                     # Streamlit-приложение (отдельный слой сервинга)
├── requirements.txt                     # Зависимости Python
└── README.md                            # Текущее описание
```

Файл `clean_mers.csv` и некоторые артефакты могут не лежать в Git и генерироваться локально.