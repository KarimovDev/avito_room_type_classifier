# EfficientNet baseline

Минимальный baseline для обучения EfficientNet на общем data pipeline проекта.

## B0 или B1

EfficientNet-B1 больше, чем B0: у нее выше входное разрешение (240) в оригинальной
семье моделей, больше параметров и вычислений. На практике B1 может дать лучшее
качество, но обучается и инференсится медленнее. Попробуем оба варианта.

Размер входа и тип модели можно менять:

```bash
just run --group efficientnet python -m models.efficientNet.train_efficientnet --variant b1 --image-size 240
```

## Запуск

```bash
just install-efficientnet
just prepare-data
just train-efficientnet
```

```bash
just run --group efficientnet python -m models.efficientNet.train_efficientnet
```

По умолчанию обучение читает `data/processed/train_df.csv` и
`data/processed/val_df.csv`. Эти файлы создаются командой `just prepare-data`:
класс `18` удаляется, старый класс `19` становится новым классом `18`, поэтому
модель обучается на 19 классах.

Дисбаланс классов учитывается внутри обучения через веса классов в

```bash
just run --group efficientnet python -m models.efficientNet.train_efficientnet --class-balance none
```

Можно включить балансировку на уровне DataLoader через `WeightedRandomSampler`:

```bash
just run --group efficientnet python -m models.efficientNet.train_efficientnet --use-weighted-sampling --class-balance none
```

## Результаты

Скрипт сохраняет:

```text
outputs/models/efficientnet/efficientnet_b0_best.pt
reports/metrics/efficientnet/efficientnet_b0_metrics.json
reports/metrics/efficientnet/model_comparison.csv
```

Основная метрика для ТЗ: `best_macro_f1`.
В конце обучения будет информация f1 в разрезе каждого класса `best_per_class_f1` внутри metrics JSON.

`model_comparison.csv` можно использовать как простую таблицу сравнения с
запусков и моделей, добавляя туда строки с их результатами.

## Grad-CAM

Построить Grad-CAM для первого доступного примера из validation:

```bash
just install-interpretability
just grad-cam-efficientnet
```

Для конкретного изображения:

```bash
just run --group efficientnet --group interpretability python models/efficientNet/grad_cam.py --checkpoint outputs/models/efficientnet/efficientnet_b1_best.pt --image data/raw/val_images/14333332896.jpg
```

Результаты сохраняются в:

```text
outputs/grad_cam/efficientnet/
```
