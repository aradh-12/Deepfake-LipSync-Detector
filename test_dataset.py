from utils.dataset_manager import get_all_videos

videos = get_all_videos()

print(f"Total videos: {len(videos)}")

for video, label in videos[:10]:
    print(label, video.name)