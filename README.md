# somethingsomething
Fork from https://github.com/joaanna/something_else

### Add Dataset
Download videos from the dataset provider:
https://20bn.com/datasets/something-something
Download and unzip all videos to `/something_videos` under the root directory of the project. (The folder does not come with this repo, create them by yourself.)

### Download Bounding box annotation JSON
https://drive.google.com/drive/folders/1XqZC2jIHqrLPugPOVJxCH_YWa275PBrZ
Download 4 JSONs files from the google drive, and put them under `bounding_box_annotations` folder (The folder does not come with this repo, create them by yourself.)

### Extract frames with 12 FPS from the videos
The original experiments did not provide the code to extract frames. 
I created fps.extraction.py to extract frame, and examine the original .webm has fps 12 by cv2 examination. 
However the frame number does not match theirs. (The frame number doesn't match their annotations.)
Is trying another method in this repo:
https://github.com/facebookresearch/SlowFast/blob/master/slowfast/datasets/DATASET.md#something-something-v2
