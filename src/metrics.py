import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def calculate_accuracy(y_true, y_pred):
    """Считает accuracy: долю правильно предсказанных классов."""
    # Преобразуем предсказанные и реальные классы в numpy-массивы
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return accuracy_score(
        y_true=y_true,
        y_pred=y_pred,
    )


def calculate_macro_f1(y_true, y_pred):
    # Преобразуем предсказанные и реальные классы в numpy-массивы
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return f1_score(
        y_true=y_true,
        y_pred=y_pred,
        average='macro'
    )


def calculate_per_class_f1(y_true, y_pred, num_classes):
    # Считаем F1 отдельно для каждой числовой метки от 0 до num_classes - 1.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))

    scores = f1_score(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    supports = np.bincount(y_true, minlength=num_classes)

    return [
        {
            "class_id": class_id,
            "f1": float(scores[class_id]),
            "support": int(supports[class_id]),
        }
        for class_id in labels
    ]
