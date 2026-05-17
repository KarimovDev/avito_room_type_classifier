# Room Type Classifier

Проект по классификации типа комнаты по изображению
Кейс: Avito, определение одного из 19 типов помещений по фотографии
Основная метрика качества: Macro F1
Лучший одиночный вариант: ConvNeXt Nano
Итоговое решение для submission: ансамбль ConvNeXt Nano, ResNet50 и ResNet18

## Что сделано

- подготовлен preprocessing для train, val и test
- удален лишний train-класс, которого нет в задании и validation
- классы приведены к диапазону 0-18
- обучены несколько CNN-моделей
- эксперименты залогированы в MLflow через DagsHub
- собран итоговый ансамбль из трех моделей
- подготовлен submission на test
- сделан Streamlit-прототип для проверки изображений

## Данные

Raw-данные ожидаются в таком виде:

```text
data/raw/
  train_df.csv
  val_df.csv
  test_df.csv
  train_images/
  val_images/
  test_images/
```

Препроцессинг:

```bash
just prepare-data-with-heuristics
```

Processed-файлы сохраняются в:

```text
data/processed/
```

## Установка

Нужны Python 3.12, `uv` и `just`.

```bash
pip install uv
uv tool install rust-just
```

Установа зависимостей:

```bash
just install-all
```

Установка зависимостей для Streamlit:

```bash
just install-streamlit
```

## Обучение

Примеры команд:

```bash
just train-resnet18 30
just train-resnet50 30 32
just train-efficientnet-b0 30 32
just train-efficientnet-b1 30 32
just train-densenet121 2 8 5 32
just train-convnext-nano 25 32
just train-convnext-tiny
```

Сравнение моделей(MLflow):

```bash
just compare-models
```

Оценка итогового ансамбля:

```bash
just eval-ensemble
```

## Submission

Генерация submission:

```bash
just make-submission
```

Файл сохраняется локально:

```text
data/submissions/submission_ensemble.csv
```

## Streamlit

Запуск streamlit-интерфейса:

```bash
just run-streamlit
```

Интерфейс позволяет загрузить изображение и получить предсказания выбранных моделей. В нем доступен финальный ансамбль и отдельные модели

## Эксперименты

Эксперименты логируются в MLflow через DagsHub

Ссылка:

```text
https://dagshub.com/YashinSergey/room_type_classifier/experiments
```

Основные метрики:

- best_macro_f1
- best_accuracy
- best_train_loss
- best_val_loss
- best_epoch

## Структура проекта

```text
src/
  dataset.py
  dataloaders.py
  preprocess_data.py
  transforms.py
  infer_ensemble.py
  evaluate_ensemble.py

models/
  convnext_nano/
  convnext_tiny/
  densenet121/
  efficientNet/
  resnet18/
  resnet50/
  yolo/

streamlit/
  app.py

reports/metrics/
outputs/models/
data/
```

Чекпоинты моделей сохраняются в `outputs/models/`.

Метрики сохраняются в `reports/metrics/`.

Итоговый отчет: https://docs.google.com/document/d/1LT4T90vRei1lcjus16heISa5xYsl4cUo/edit?usp=sharing&ouid=103046931125072858111&rtpof=true&sd=true

Презентация: https://docs.google.com/presentation/d/1Wa8_ZrLCPGx9kds4LiGtT-41lT_W1HoH/edit?usp=sharing&ouid=103046931125072858111&rtpof=true&sd=true
