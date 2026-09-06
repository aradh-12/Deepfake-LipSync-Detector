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

PREDICTIONS_PATH = Path(
    "saved_models/lstm_exp14_test_predictions.npz"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n" + "=" * 85
    )

    print(
        "EXP14 VIDEO THRESHOLD DIAGNOSTIC"
    )

    print(
        "=" * 85
    )


    # ========================================================
    # CHECK FILE
    # ========================================================

    if not PREDICTIONS_PATH.exists():

        raise FileNotFoundError(

            f"Predictions not found: "
            f"{PREDICTIONS_PATH}"

        )


    # ========================================================
    # LOAD PREDICTIONS
    # ========================================================

    data = np.load(

        PREDICTIONS_PATH,

        allow_pickle=True

    )


    video_probabilities = data[
        "video_probabilities"
    ]


    video_labels = data[
        "video_labels"
    ]


    video_names = data[
        "video_names"
    ]


    # ========================================================
    # THRESHOLDS
    # ========================================================

    thresholds = [

        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.44,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80

    ]


    print(
        "\n"
        "Threshold | Accuracy | Precision | Recall | "
        "F1 | FP | FN"
    )


    print(
        "-" * 85
    )


    # ========================================================
    # TEST EACH THRESHOLD
    # ========================================================

    for threshold in thresholds:


        predictions = (

            video_probabilities >= threshold

        ).astype(

            np.int32

        )


        accuracy = accuracy_score(

            video_labels,

            predictions

        )


        precision = precision_score(

            video_labels,

            predictions,

            zero_division=0

        )


        recall = recall_score(

            video_labels,

            predictions,

            zero_division=0

        )


        f1 = f1_score(

            video_labels,

            predictions,

            zero_division=0

        )


        matrix = confusion_matrix(

            video_labels,

            predictions,

            labels=[0, 1]

        )


        fp = matrix[0, 1]

        fn = matrix[1, 0]


        print(

            f"{threshold:9.2f} | "

            f"{accuracy * 100:7.2f}% | "

            f"{precision * 100:8.2f}% | "

            f"{recall * 100:6.2f}% | "

            f"{f1 * 100:6.2f}% | "

            f"{fp:2d} | "

            f"{fn:2d}"

        )


    # ========================================================
    # VIDEO PROBABILITY DISTRIBUTION
    # ========================================================

    print(
        "\n" + "=" * 85
    )

    print(
        "VIDEO PROBABILITY DISTRIBUTION"
    )

    print(
        "=" * 85
    )


    real_probabilities = (

        video_probabilities[
            video_labels == 0
        ]

    )


    fake_probabilities = (

        video_probabilities[
            video_labels == 1
        ]

    )


    print(
        "\nREAL VIDEOS"
    )


    for probability in sorted(

        real_probabilities

    ):

        print(

            f"{probability:.6f}"

        )


    print(
        "\nFAKE VIDEOS"
    )


    for probability in sorted(

        fake_probabilities

    ):

        print(

            f"{probability:.6f}"

        )


    # ========================================================
    # VIDEO-WISE PROBABILITIES
    # ========================================================

    print(
        "\n" + "=" * 85
    )

    print(
        "VIDEO-WISE PROBABILITIES"
    )

    print(
        "=" * 85
    )


    for (

        video_name,
        label,
        probability

    ) in zip(

        video_names,
        video_labels,
        video_probabilities

    ):


        actual = (

            "FAKE"

            if label == 1

            else "REAL"

        )


        print()

        print(
            video_name
        )


        print(
            "Actual:",
            actual
        )


        print(
            "Probability:",
            round(
                float(probability),
                6
            )
        )


    print(
        "\n" + "=" * 85
    )

    print(
        "ANALYSIS COMPLETED"
    )

    print(
        "=" * 85
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()