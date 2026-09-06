from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path(
    "outputs/dataset_exp8_temporal"
)

MODEL_PATH = Path(
    "saved_models/lstm_exp9.keras"
)

NORMALIZATION_PATH = Path(
    "saved_models/lstm_exp9_normalization.npz"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(split_name):

    file_path = (
        DATASET_DIR /
        f"{split_name}.npz"
    )

    data = np.load(
        file_path,
        allow_pickle=True
    )

    X = data["X"].astype(
        np.float32
    )

    y = data["y"].astype(
        np.int32
    )

    video_ids = data[
        "video_ids"
    ]

    return X, y, video_ids


# ============================================================
# LOAD NORMALIZATION STATISTICS
# ============================================================

def load_normalization():

    data = np.load(
        NORMALIZATION_PATH
    )

    mean = data["mean"]

    std = data["std"]

    return mean, std


# ============================================================
# NORMALIZE DATA
# ============================================================

def normalize_data(
    X,
    mean,
    std
):

    return (
        X - mean
    ) / std


# ============================================================
# PRINT CLASS STATISTICS
# ============================================================

def print_class_statistics(
    probabilities,
    labels,
    class_value,
    class_name
):

    class_probabilities = (
        probabilities[
            labels == class_value
        ]
    )

    print()
    print(
        f"{class_name} Samples: "
        f"{len(class_probabilities)}"
    )

    if len(class_probabilities) == 0:

        print(
            "No samples available."
        )

        return

    print(
        f"Mean   : "
        f"{np.mean(class_probabilities):.4f}"
    )

    print(
        f"Median : "
        f"{np.median(class_probabilities):.4f}"
    )

    print(
        f"Std    : "
        f"{np.std(class_probabilities):.4f}"
    )

    print(
        f"Minimum: "
        f"{np.min(class_probabilities):.4f}"
    )

    print(
        f"Maximum: "
        f"{np.max(class_probabilities):.4f}"
    )


# ============================================================
# ANALYZE ONE SPLIT
# ============================================================

def analyze_split(
    split_name,
    model,
    mean,
    std
):

    print()
    print("=" * 70)

    print(
        f"{split_name.upper()} "
        f"PROBABILITY ANALYSIS"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print()

    print(
        "Loading dataset..."
    )

    X, y, video_ids = (
        load_dataset(
            split_name
        )
    )

    print(
        "Sequences:",
        len(X)
    )

    print(
        "Videos:",
        len(
            np.unique(video_ids)
        )
    )

    print(
        "Real sequences:",
        np.sum(y == 0)
    )

    print(
        "Fake sequences:",
        np.sum(y == 1)
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    print()

    print(
        "Normalizing..."
    )

    X = normalize_data(
        X,
        mean,
        std
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    print()

    print(
        "Running predictions..."
    )

    probabilities = (
        model.predict(
            X,
            verbose=0
        )
        .flatten()
    )

    # --------------------------------------------------------
    # OVERALL PROBABILITY STATISTICS
    # --------------------------------------------------------

    print()
    print("-" * 70)

    print(
        "OVERALL PROBABILITY STATISTICS"
    )

    print("-" * 70)

    print(
        f"Mean   : "
        f"{np.mean(probabilities):.4f}"
    )

    print(
        f"Median : "
        f"{np.median(probabilities):.4f}"
    )

    print(
        f"Std    : "
        f"{np.std(probabilities):.4f}"
    )

    print(
        f"Minimum: "
        f"{np.min(probabilities):.4f}"
    )

    print(
        f"Maximum: "
        f"{np.max(probabilities):.4f}"
    )

    # --------------------------------------------------------
    # REAL STATISTICS
    # --------------------------------------------------------

    print()
    print("-" * 70)

    print(
        "REAL PROBABILITY STATISTICS"
    )

    print("-" * 70)

    print_class_statistics(
        probabilities,
        y,
        0,
        "Real"
    )

    # --------------------------------------------------------
    # FAKE STATISTICS
    # --------------------------------------------------------

    print()
    print("-" * 70)

    print(
        "FAKE PROBABILITY STATISTICS"
    )

    print("-" * 70)

    print_class_statistics(
        probabilities,
        y,
        1,
        "Fake"
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    print()
    print("-" * 70)

    print(
        "ROC-AUC ANALYSIS"
    )

    print("-" * 70)

    if len(np.unique(y)) == 2:

        auc_score = roc_auc_score(
            y,
            probabilities
        )

        print(
            f"Sequence-level ROC-AUC: "
            f"{auc_score:.4f}"
        )

    else:

        print(
            "Cannot calculate ROC-AUC."
        )

        auc_score = None

    # --------------------------------------------------------
    # VIDEO-LEVEL PROBABILITIES
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "VIDEO-LEVEL PROBABILITY ANALYSIS"
    )

    print("=" * 70)

    unique_videos = (
        np.unique(video_ids)
    )

    video_probabilities = []

    video_labels = []

    for video_id in unique_videos:

        indices = np.where(
            video_ids == video_id
        )[0]

        video_probability = (
            np.mean(
                probabilities[
                    indices
                ]
            )
        )

        video_label = (
            y[indices[0]]
        )

        video_probabilities.append(
            video_probability
        )

        video_labels.append(
            video_label
        )

    video_probabilities = np.asarray(
        video_probabilities
    )

    video_labels = np.asarray(
        video_labels
    )

    print()

    print(
        "Video count:",
        len(unique_videos)
    )

    # --------------------------------------------------------
    # VIDEO REAL STATISTICS
    # --------------------------------------------------------

    print()
    print(
        "VIDEO-LEVEL REAL "
        "PROBABILITY STATISTICS"
    )

    print_class_statistics(
        video_probabilities,
        video_labels,
        0,
        "Real"
    )

    # --------------------------------------------------------
    # VIDEO FAKE STATISTICS
    # --------------------------------------------------------

    print()
    print(
        "VIDEO-LEVEL FAKE "
        "PROBABILITY STATISTICS"
    )

    print_class_statistics(
        video_probabilities,
        video_labels,
        1,
        "Fake"
    )

    # --------------------------------------------------------
    # VIDEO ROC-AUC
    # --------------------------------------------------------

    print()

    if len(np.unique(video_labels)) == 2:

        video_auc_score = (
            roc_auc_score(
                video_labels,
                video_probabilities
            )
        )

        print(
            f"\nVideo-level ROC-AUC: "
            f"{video_auc_score:.4f}"
        )

    else:

        print(
            "\nCannot calculate "
            "video-level ROC-AUC."
        )

        video_auc_score = None

    # --------------------------------------------------------
    # OVERLAP ANALYSIS
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "CLASS PROBABILITY OVERLAP"
    )

    print("=" * 70)

    real_probabilities = (
        probabilities[y == 0]
    )

    fake_probabilities = (
        probabilities[y == 1]
    )

    if (
        len(real_probabilities) > 0
        and len(fake_probabilities) > 0
    ):

        real_min = np.min(
            real_probabilities
        )

        real_max = np.max(
            real_probabilities
        )

        fake_min = np.min(
            fake_probabilities
        )

        fake_max = np.max(
            fake_probabilities
        )

        overlap_min = max(
            real_min,
            fake_min
        )

        overlap_max = min(
            real_max,
            fake_max
        )

        print()

        print(
            f"Real range: "
            f"{real_min:.4f} "
            f"to {real_max:.4f}"
        )

        print(
            f"Fake range: "
            f"{fake_min:.4f} "
            f"to {fake_max:.4f}"
        )

        print()

        if overlap_min <= overlap_max:

            print(
                f"Probability overlap: "
                f"{overlap_min:.4f} "
                f"to {overlap_max:.4f}"
            )

            print()

            print(
                "RESULT: "
                "Real and Fake probabilities "
                "overlap."
            )

        else:

            print(
                "RESULT: "
                "No direct probability range "
                "overlap detected."
            )

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {
        "sequence_auc": auc_score,
        "video_auc": video_auc_score
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)

    print(
        "EXP9 PROBABILITY DISTRIBUTION ANALYSIS"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD NORMALIZATION
    # --------------------------------------------------------

    print()

    print(
        "Loading normalization statistics..."
    )

    mean, std = (
        load_normalization()
    )

    print(
        "Mean shape:",
        mean.shape
    )

    print(
        "Std shape:",
        std.shape
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()

    print(
        "Loading Exp9 model..."
    )

    model = (
        tf.keras.models.load_model(
            MODEL_PATH
        )
    )

    # --------------------------------------------------------
    # ANALYZE TRAIN
    # --------------------------------------------------------

    train_results = analyze_split(
        "train",
        model,
        mean,
        std
    )

    # --------------------------------------------------------
    # ANALYZE VALIDATION
    # --------------------------------------------------------

    validation_results = analyze_split(
        "validation",
        model,
        mean,
        std
    )

    # --------------------------------------------------------
    # ANALYZE TEST
    # --------------------------------------------------------

    test_results = analyze_split(
        "test",
        model,
        mean,
        std
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()

    print("=" * 70)

    print(
        "FINAL ROC-AUC COMPARISON"
    )

    print("=" * 70)

    print()

    print(
        "TRAIN"
    )

    print(
        f"Sequence AUC: "
        f"{train_results['sequence_auc']:.4f}"
    )

    print(
        f"Video AUC   : "
        f"{train_results['video_auc']:.4f}"
    )

    print()

    print(
        "VALIDATION"
    )

    print(
        f"Sequence AUC: "
        f"{validation_results['sequence_auc']:.4f}"
    )

    print(
        f"Video AUC   : "
        f"{validation_results['video_auc']:.4f}"
    )

    print()

    print(
        "TEST"
    )

    print(
        f"Sequence AUC: "
        f"{test_results['sequence_auc']:.4f}"
    )

    print(
        f"Video AUC   : "
        f"{test_results['video_auc']:.4f}"
    )

    print()

    print("=" * 70)

    print(
        "EXP9 PROBABILITY DISTRIBUTION "
        "ANALYSIS COMPLETED"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
