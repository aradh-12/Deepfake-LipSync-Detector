from pathlib import Path
import re


ROOT = Path("outputs/lip_coordinates")


def get_frame_number(path):
    match = re.search(r"frame_(\d+)\.csv$", path.name)

    if not match:
        return None

    return int(match.group(1))


def main():

    print("\n==============================")
    print("Lip Frame Continuity Check")
    print("==============================")

    total_videos = 0
    perfect_videos = 0
    videos_with_gaps = 0

    total_expected = 0
    total_found = 0
    total_missing = 0

    for video_folder in sorted(ROOT.iterdir()):

        if not video_folder.is_dir():
            continue

        files = sorted(
            video_folder.glob("frame_*.csv")
        )

        if not files:
            continue

        frame_numbers = sorted(
            get_frame_number(f)
            for f in files
        )

        frame_numbers = [
            x for x in frame_numbers
            if x is not None
        ]

        if not frame_numbers:
            continue

        total_videos += 1

        first_frame = min(frame_numbers)
        last_frame = max(frame_numbers)

        expected = last_frame - first_frame + 1
        found = len(frame_numbers)
        missing = expected - found

        total_expected += expected
        total_found += found
        total_missing += missing

        if missing == 0:
            perfect_videos += 1
        else:
            videos_with_gaps += 1

            print(
                f"\n⚠️ {video_folder.name}"
            )

            print(
                f"   First frame : {first_frame}"
            )

            print(
                f"   Last frame  : {last_frame}"
            )

            print(
                f"   Expected    : {expected}"
            )

            print(
                f"   Found       : {found}"
            )

            print(
                f"   Missing     : {missing}"
            )

    print("\n==============================")
    print("RESULT")
    print("==============================")

    print(
        "Videos checked      :",
        total_videos
    )

    print(
        "Perfect videos      :",
        perfect_videos
    )

    print(
        "Videos with gaps    :",
        videos_with_gaps
    )

    print(
        "Expected frames     :",
        total_expected
    )

    print(
        "Found frames        :",
        total_found
    )

    print(
        "Missing frames      :",
        total_missing
    )

    if total_expected > 0:

        detection_rate = (
            total_found /
            total_expected
            * 100
        )

        print(
            f"Detection coverage : "
            f"{detection_rate:.2f}%"
        )

    print("==============================\n")


if __name__ == "__main__":
    main()