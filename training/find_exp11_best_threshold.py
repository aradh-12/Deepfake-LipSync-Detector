from pathlib import Path

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = Path(
    "saved_models"
)

PREDICTIONS_PATH = (
    MODEL_DIR /
    "lstm_exp11_test_predictions.npz"
)


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(
        np.int32
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        predictions
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" +
        "=" * 60
    )

    print(
        "EXP11 BEST THRESHOLD ANALYSIS"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # LOAD PREDICTIONS
    # ========================================================

    if not PREDICTIONS_PATH.exists():

        raise FileNotFoundError(
            f"Predictions not found: "
            f"{PREDICTIONS_PATH}"
        )


    print(
        "\nLoading saved predictions..."
    )

    data = np.load(
        PREDICTIONS_PATH,
        allow_pickle=True
    )


    # ========================================================
    # SEQUENCE DATA
    # ========================================================

    sequence_probabilities = data[
        "sequence_probabilities"
    ]

    sequence_labels = data[
        "sequence_labels"
    ]


    # ========================================================
    # VIDEO DATA
    # ========================================================

    video_probabilities = data[
        "video_probabilities"
    ]

    video_labels = data[
        "video_labels"
    ]


    print(
        f"\nSequence samples: "
        f"{len(sequence_labels)}"
    )

    print(
        f"Video samples: "
        f"{len(video_labels)}"
    )


    # ========================================================
    # THRESHOLD RANGE
    # ========================================================

    thresholds = np.arange(
        0.05,
        0.96,
        0.01
    )


    # ========================================================
    # SEQUENCE LEVEL SEARCH
    # ========================================================

    print(
        "\n" +
        "=" * 60
    )

    print(
        "SEQUENCE-LEVEL THRESHOLD SEARCH"
    )

    print(
        "=" * 60
    )


    best_sequence_f1 = -1

    best_sequence_threshold = None

    best_sequence_results = None


    for threshold in thresholds:

        (
            accuracy,
            precision,
            recall,
            f1,
            predictions

        ) = calculate_metrics(

            sequence_labels,

            sequence_probabilities,

            threshold
        )


        if f1 > best_sequence_f1:

            best_sequence_f1 = f1

            best_sequence_threshold = threshold

            best_sequence_results = (

                accuracy,

                precision,

                recall,

                f1,

                predictions
            )


    (
        accuracy,
        precision,
        recall,
        f1,
        predictions

    ) = best_sequence_results


    print(
        f"\nBest threshold: "
        f"{best_sequence_threshold:.2f}"
    )

    print(
        f"Accuracy       : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision      : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall         : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score       : "
        f"{f1 * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            sequence_labels,
            predictions
        )
    )


    # ========================================================
    # VIDEO LEVEL SEARCH
    # ========================================================

    print(
        "\n" +
        "=" * 60
    )

    print(
        "VIDEO-LEVEL THRESHOLD SEARCH"
    )

    print(
        "=" * 60
    )


    best_video_f1 = -1

    best_video_threshold = None

    best_video_results = None


    for threshold in thresholds:

        (
            accuracy,
            precision,
            recall,
            f1,
            predictions

        ) = calculate_metrics(

            video_labels,

            video_probabilities,

            threshold
        )


        if f1 > best_video_f1:

            best_video_f1 = f1

            best_video_threshold = threshold

            best_video_results = (

                accuracy,

                precision,

                recall,

                f1,

                predictions
            )


    (
        accuracy,
        precision,
        recall,
        f1,
        predictions

    ) = best_video_results


    print(
        f"\nBest threshold: "
        f"{best_video_threshold:.2f}"
    )

    print(
        f"Accuracy       : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision      : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall         : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1 Score       : "
        f"{f1 * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_matrix(
            video_labels,
            predictions
        )
    )


    # ========================================================
    # THRESHOLD 0.50 COMPARISON
    # ========================================================

    print(
        "\n" +
        "=" * 60
    )

    print(
        "COMPARISON WITH DEFAULT THRESHOLD 0.50"
    )

    print(
        "=" * 60
    )


    print(
        "\nSEQUENCE LEVEL"
    )

    for threshold in [

        0.50,

        best_sequence_threshold

    ]:

        (
            accuracy,
            precision,
            recall,
            f1,
            _

        ) = calculate_metrics(

            sequence_labels,

            sequence_probabilities,

            threshold
        )


        print(
            f"\nThreshold: "
            f"{threshold:.2f}"
        )

        print(
            f"Accuracy : "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"Precision: "
            f"{precision * 100:.2f}%"
        )

        print(
            f"Recall   : "
            f"{recall * 100:.2f}%"
        )

        print(
            f"F1 Score : "
            f"{f1 * 100:.2f}%"
        )


    print(
        "\nVIDEO LEVEL"
    )

    for threshold in [

        0.50,

        best_video_threshold

    ]:

        (
            accuracy,
            precision,
            recall,
            f1,
            _

        ) = calculate_metrics(

            video_labels,

            video_probabilities,

            threshold
        )


        print(
            f"\nThreshold: "
            f"{threshold:.2f}"
        )

        print(
            f"Accuracy : "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"Precision: "
            f"{precision * 100:.2f}%"
        )

        print(
            f"Recall   : "
            f"{recall * 100:.2f}%"
        )

        print(
            f"F1 Score : "
            f"{f1 * 100:.2f}%"
        )


    print(
        "\n" +
        "=" * 60
    )

    print(
        "EXP11 BEST THRESHOLD ANALYSIS COMPLETED"
    )

    print(
        "=" * 60 +
        "\n"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()