# somethingsomething
Fork from https://github.com/joaanna/something_else

### Add Dataset
Download videos from the dataset provider:
https://20bn.com/datasets/something-something
Download and unzip all videos to `/something_videos` under the root directory of the project. (The folder does not come with this repo, create them by yourself.)

### Download Bounding box annotation JSON
https://drive.google.com/drive/folders/1XqZC2jIHqrLPugPOVJxCH_YWa275PBrZ
Download 4 JSONs files from the google drive, and put them under `/bounding_box_annotations` folder (The folder does not come with this repo, create them by yourself.)

### Extract frames with 12 FPS from the videos
1. Install `ffmpeg` by running `sudo apt-get install ffmpeg` in terminal  (resource: https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu)
2. Run `batch_fps_conversion.sh` script to extract all videos (under `/something_videos`) into frames extracted by fps 12 (frame per second rate).
3. See the result in `/something_videos_frames`. Every video frames are extracted into the folder, named after each video's basename.


