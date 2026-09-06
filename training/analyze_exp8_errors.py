from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = Path(
    "outputs/dataset_exp8_temporal/test.npz"
)

MODEL_FILE = Path(
    "saved_models/lstm_exp8.keras"
)

NORMALIZATION_FILE = Path(
    "saved_models/lstm_exp8_normalization.npz"
)

THRESHOLD = 0.65


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("EXP8 TEST ERROR ANALYSIS")
print("=" * 70)

print(f"\nDataset file:")
print(DATASET_FILE)

print(f"\nModel file:")
print(MODEL_FILE)

print(f"\nNormalization file:")
print(NORMALIZATION_FILE)

print(f"\nThreshold: {THRESHOLD}")


# ============================================================
# CHECK FILES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING REQUIRED FILES")
print("=" * 70)


if not DATASET_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset file not found:\n{DATASET_FILE}"
    )


if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"\nModel file not found:\n{MODEL_FILE}"
    )


if not NORMALIZATION_FILE.exists():

    raise FileNotFoundError(
        f"\nNormalization file not found:\n"
        f"{NORMALIZATION_FILE}"
    )


print("\nAll required files found successfully.")


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)


data = np.load(
    DATASET_FILE,
    allow_pickle=True
)


X = data["X"].astype(np.float32)

y = data["y"].astype(np.int32)

video_ids = data["video_ids"]


print(f"\nTest sequences: {len(X)}")

print(f"Feature shape: {X.shape}")

print(f"Real sequences: {np.sum(y == 0)}")

print(f"Fake sequences: {np.sum(y == 1)}")


unique_videos = np.unique(
    video_ids
)


print(f"Unique videos: {len(unique_videos)}")


# ============================================================
# LOAD NORMALIZATION
# ============================================================

print("\n" + "=" * 70)
print("LOADING NORMALIZATION STATISTICS")
print("=" * 70)


normalization = np.load(
    NORMALIZATION_FILE
)


mean = normalization["mean"]

std = normalization["std"]


print(f"\nMean shape: {mean.shape}")

print(f"Std shape: {std.shape}")


# ============================================================
# NORMALIZE DATA
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZING TEST DATA")
print("=" * 70)


X_normalized = (
    X - mean
) / (
    std + 1e-8
)


print("\nNormalization completed successfully.")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING EXP8 MODEL")
print("=" * 70)


model = tf.keras.models.load_model(
    MODEL_FILE
)


print("\nModel loaded successfully.")


# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("RUNNING PREDICTIONS")
print("=" * 70)


sequence_probabilities = model.predict(
    X_normalized,
    verbose=0
).reshape(-1)


sequence_predictions = (
    sequence_probabilities >= THRESHOLD
).astype(int)


print("\nPredictions completed successfully.")


# ============================================================
# SEQUENCE-LEVEL ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("SEQUENCE-LEVEL ERROR ANALYSIS")
print("=" * 70)


correct_sequences = (
    sequence_predictions == y
)


incorrect_sequences = (
    sequence_predictions != y
)


print(f"\nCorrect sequences: "
      f"{np.sum(correct_sequences)}")

print(f"Incorrect sequences: "
      f"{np.sum(incorrect_sequences)}")


false_positive_indices = np.where(
    (y == 0)
    &
    (sequence_predictions == 1)
)[0]


false_negative_indices = np.where(
    (y == 1)
    &
    (sequence_predictions == 0)
)[0]


print(f"\nFalse Positives: "
      f"{len(false_positive_indices)}")

print(
    "Real sequences predicted as Fake"
)


print(f"\nFalse Negatives: "
      f"{len(false_negative_indices)}")

print(
    "Fake sequences predicted as Real"
)


# ============================================================
# VIDEO-LEVEL AGGREGATION
# ============================================================

print("\n" + "=" * 70)
print("VIDEO-LEVEL AGGREGATION")
print("=" * 70)


video_probabilities = []

video_labels = []

video_names = []


for video_id in unique_videos:

    indices = np.where(
        video_ids == video_id
    )[0]


    probabilities = sequence_probabilities[
        indices
    ]


    labels = y[
        indices
    ]


    # --------------------------------------------------------
    # Average probability across sequences
    # --------------------------------------------------------

    video_probability = np.mean(
        probabilities
    )


    # --------------------------------------------------------
    # Get video label
    # --------------------------------------------------------

    unique_labels = np.unique(
        labels
    )


    if len(unique_labels) != 1:

        raise RuntimeError(
            f"\nVideo has multiple labels:\n"
            f"{video_id}\n"
            f"Labels: {unique_labels}"
        )


    video_label = int(
        unique_labels[0]
    )


    video_probabilities.append(
        video_probability
    )


    video_labels.append(
        video_label
    )


    video_names.append(
        video_id
    )


video_probabilities = np.array(
    video_probabilities
)


video_labels = np.array(
    video_labels
)


video_predictions = (
    video_probabilities >= THRESHOLD
).astype(int)


print(f"\nTotal videos: "
      f"{len(video_names)}")

print(
    f"Real videos: "
    f"{np.sum(video_labels == 0)}"
)

print(
    f"Fake videos: "
    f"{np.sum(video_labels == 1)}"
)


# ============================================================
# VIDEO-LEVEL ERRORS
# ============================================================

print("\n" + "=" * 70)
print("VIDEO-LEVEL ERROR ANALYSIS")
print("=" * 70)


correct_videos = (
    video_predictions == video_labels
)


incorrect_videos = (
    video_predictions != video_labels
)


print(
    f"\nCorrect videos: "
    f"{np.sum(correct_videos)}"
)


print(
    f"Incorrect videos: "
    f"{np.sum(incorrect_videos)}"
)


# ============================================================
# FALSE POSITIVES
# ============================================================

print("\n" + "=" * 70)
print("FALSE POSITIVES")
print("REAL VIDEOS PREDICTED AS FAKE")
print("=" * 70)


false_positive_videos = np.where(
    (video_labels == 0)
    &
    (video_predictions == 1)
)[0]


if len(false_positive_videos) == 0:

    print("\nNo False Positives.")

else:

    for index in false_positive_videos:

        print(f"\nVideo: {video_names[index]}")

        print("Actual: Real")

        print(
            f"Probability: "
            f"{video_probabilities[index]:.4f}"
        )

        print("Predicted: Fake")


# ============================================================
# FALSE NEGATIVES
# ============================================================

print("\n" + "=" * 70)
print("FALSE NEGATIVES")
print("FAKE VIDEOS PREDICTED AS REAL")
print("=" * 70)


false_negative_videos = np.where(
    (video_labels == 1)
    &
    (video_predictions == 0)
)[0]


if len(false_negative_videos) == 0:

    print("\nNo False Negatives.")

else:

    for index in false_negative_videos:

        print(f"\nVideo: {video_names[index]}")

        print("Actual: Fake")

        print(
            f"Probability: "
            f"{video_probabilities[index]:.4f}"
        )

        print("Predicted: Real")


# ============================================================
# CORRECT PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("CORRECT VIDEO PREDICTIONS")
print("=" * 70)


for index in range(
    len(video_names)
):

    if correct_videos[index]:

        actual = (
            "Fake"
            if video_labels[index] == 1
            else "Real"
        )


        predicted = (
            "Fake"
            if video_predictions[index] == 1
            else "Real"
        )


        print(f"\nVideo: {video_names[index]}")

        print(f"Actual: {actual}")

        print(
            f"Probability: "
            f"{video_probabilities[index]:.4f}"
        )

        print(
            f"Predicted: {predicted}"
        )


# ============================================================
# PROBABILITY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("VIDEO PROBABILITY ANALYSIS")
print("=" * 70)


real_probabilities = video_probabilities[
    video_labels == 0
]


fake_probabilities = video_probabilities[
    video_labels == 1
]


print("\nREAL VIDEOS")

print(
    f"Mean probability: "
    f"{np.mean(real_probabilities):.4f}"
)

print(
    f"Minimum probability: "
    f"{np.min(real_probabilities):.4f}"
)

print(
    f"Maximum probability: "
    f"{np.max(real_probabilities):.4f}"
)


print("\nFAKE VIDEOS")

print(
    f"Mean probability: "
    f"{np.mean(fake_probabilities):.4f}"
)

print(
    f"Minimum probability: "
    f"{np.min(fake_probabilities):.4f}"
)

print(
    f"Maximum probability: "
    f"{np.max(fake_probabilities):.4f}"
)


# ============================================================
# ERROR SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ERROR SUMMARY")
print("=" * 70)


print(
    f"\nTotal videos: "
    f"{len(video_names)}"
)


print(
    f"Correct predictions: "
    f"{np.sum(correct_videos)}"
)


print(
    f"Incorrect predictions: "
    f"{np.sum(incorrect_videos)}"
)


print(
    f"\nFalse Positives: "
    f"{len(false_positive_videos)}"
)


print(
    f"False Negatives: "
    f"{len(false_negative_videos)}"
)


print(
    "\nInterpretation:"
)


if len(false_positive_videos) > len(false_negative_videos):

    print(
        "The model is producing more "
        "False Positives."
    )

elif len(false_negative_videos) > len(false_positive_videos):

    print(
        "The model is producing more "
        "False Negatives."
    )

else:

    print(
        "False Positives and False Negatives "
        "are balanced."
    )


print("\n" + "=" * 70)
print("EXP8 TEST ERROR ANALYSIS COMPLETED")
print("=" * 70)