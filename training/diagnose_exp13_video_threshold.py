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

PREDICTION_FILE = Path(
    "saved_models/lstm_exp13_test_predictions.npz"
)


# ============================================================
# THRESHOLDS TO TEST
# ============================================================

THRESHOLDS = [

    0.20,
    0.25,
    0.28,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70

]


# ============================================================
# MAIN
# ============================================================

def main():


    print(
        "\n" + "=" * 85
    )

    print(
        "EXP13 VIDEO THRESHOLD DIAGNOSTIC"
    )

    print(
        "=" * 85
    )


    # ========================================================
    # CHECK FILE
    # ========================================================

    if not PREDICTION_FILE.exists():

        raise FileNotFoundError(

            f"Prediction file not found: "
            f"{PREDICTION_FILE}"

        )


    # ========================================================
    # LOAD PREDICTIONS
    # ========================================================

    print(
        "\nLoading prediction file..."
    )


    data = np.load(

        PREDICTION_FILE,

        allow_pickle=True

    )


    # ========================================================
    # LOAD VIDEO DATA
    # ========================================================

    video_probabilities = data[

        "video_probabilities"

    ].astype(

        np.float32

    )


    video_labels = data[

        "video_labels"

    ].astype(

        np.int32

    )


    video_names = data[

        "video_names"

    ]


    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    print(
        "\nVideo count:",
        len(video_labels)
    )


    print(
        "Real videos:",
        np.sum(video_labels == 0)
    )


    print(
        "Fake videos:",
        np.sum(video_labels == 1)
    )


    # ========================================================
    # THRESHOLD TABLE
    # ========================================================

    print(
        "\n" + "=" * 85
    )

    print(
        "THRESHOLD RESULTS"
    )

    print(
        "=" * 85
    )


    print(

        "\nThreshold | Accuracy | Precision | Recall | "
        "F1 | FP | FN"

    )


    print(
        "-" * 85
    )


    for threshold in THRESHOLDS:


        # ----------------------------------------------------
        # CREATE PREDICTIONS
        # ----------------------------------------------------

        predictions = (

            video_probabilities >= threshold

        ).astype(

            np.int32

        )


        # ----------------------------------------------------
        # CALCULATE METRICS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        cm = confusion_matrix(

            video_labels,

            predictions,

            labels=[0, 1]

        )


        tn = cm[0, 0]

        fp = cm[0, 1]

        fn = cm[1, 0]

        tp = cm[1, 1]


        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

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
    # PROBABILITY DISTRIBUTION
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


    # ========================================================
    # SEPARATE REAL AND FAKE
    # ========================================================

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


    # ========================================================
    # PRINT REAL PROBABILITIES
    # ========================================================

    print(
        "\nREAL VIDEOS"
    )


    for probability in sorted(

        real_probabilities

    ):

        print(

            f"{probability:.6f}"

        )


    # ========================================================
    # PRINT FAKE PROBABILITIES
    # ========================================================

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
    # DETAILED VIDEO RESULTS
    # ========================================================

    print(
        "\n" + "=" * 85
    )

    print(
        "VIDEO DETAILS"
    )

    print(
        "=" * 85
    )


    for i in range(

        len(video_names)

    ):


        actual = (

            "FAKE"

            if video_labels[i] == 1

            else "REAL"

        )


        print(

            "\nVideo:",
            video_names[i]

        )


        print(

            "Actual:",
            actual

        )


        print(

            "Probability:",
            round(

                float(
                    video_probabilities[i]
                ),

                6

            )

        )


    print(
        "\n" + "=" * 85
    )

    print(
        "DIAGNOSTIC COMPLETED"
    )

    print(
        "=" * 85
    )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()