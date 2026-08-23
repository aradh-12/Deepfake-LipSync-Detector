from utils.multidataset_manager import get_all_videos

videos = get_all_videos()

print("Total:", len(videos))

for video, label in videos[:20]:
    print(label, video)