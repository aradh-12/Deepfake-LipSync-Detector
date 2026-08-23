from pathlib import Path

from feature_fusion.feature_sync import process_video

BAD_VIDEOS = [
    "FakeAVCeleb_id01528_00017",
    "FakeAVCeleb_id01530_00002",
    "FakeAVCeleb_id01544_00044",
    "FakeAVCeleb_id01597_00005",
    "FakeAVCeleb_id01598_00044",
    "FakeAVCeleb_id01610_00090",
    "FakeAVCeleb_id01637_00002",
    "FakeAVCeleb_id01691_00045",
    "FakeAVCeleb_id01717_00005",
    "FakeAVCeleb_id01779_00010",
    "FakeAVCeleb_id01835_00130",
    "FakeAVCeleb_id01856_00006",
    "FakeAVCeleb_id01920_00099",
    "FakeAVCeleb_id01933_00028",
    "FakeAVCeleb_id01972_00078",
    "FakeAVCeleb_id01995_00071",
    "FakeAVCeleb_id02005_00052",
    "FakeAVCeleb_id02040_00476",
    "FakeAVCeleb_id02051_00015",
    "FakeAVCeleb_id02268_00036",
    "FakeAVCeleb_id02316_00094",
    "FakeAVCeleb_id02342_00191",
    "FakeAVCeleb_id02494_00050",
    "FakeAVCeleb_id04727_00007",
]


def get_video_name(video):
    return (
        f"FakeAVCeleb_"
        f"{video.parent.name}_"
        f"{video.stem}"
    )


print("==============================")
print("REGENERATING BAD FEATURES")
print("==============================")


found = 0
success = 0
failed = 0


for video in Path(".").rglob("*.mp4"):

    video_name = get_video_name(video)

    if video_name not in BAD_VIDEOS:
        continue

    found += 1

    print("\n==============================")
    print(f"VIDEO {found}/{len(BAD_VIDEOS)}")
    print(video_name)
    print(video)
    print("==============================")

    try:

        shape = process_video(
            video_name,
            video
        )

        print(
            f"✅ Regenerated: {shape}"
        )

        success += 1

    except Exception as error:

        print(
            f"❌ FAILED: {video_name}"
        )

        print(
            f"Error: {error}"
        )

        failed += 1


print("\n==============================")
print("REGENERATION COMPLETE")
print("==============================")

print("Found   :", found)
print("Success :", success)
print("Failed  :", failed)

print("==============================")
