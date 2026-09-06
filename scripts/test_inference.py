import sys
from pathlib import Path


# Project root is one level above the scripts directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow importing modules from the project root
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.predict_video import predict_video


def main():
    if len(sys.argv) != 2:
        print("\nUsage:")
        print("python scripts/test_inference.py <path_to_video>\n")
        sys.exit(1)

    video_path = Path(sys.argv[1])

    if not video_path.exists():
        print(f"\nError: Video file not found: {video_path}\n")
        sys.exit(1)

    if not video_path.is_file():
        print(f"\nError: The provided path is not a file: {video_path}\n")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("DEEPFAKE LIP-SYNC INFERENCE TEST")
    print("=" * 60)
    print(f"\nVideo: {video_path}\n")

    try:
        result = predict_video(str(video_path))

        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)

        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(
            f"Real Probability: "
            f"{result['real_probability'] * 100:.2f}%"
        )
        print(
            f"Fake Probability: "
            f"{result['fake_probability'] * 100:.2f}%"
        )

        print()

    except Exception as error:
        print("\nInference failed.")
        print(f"Error: {error}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()