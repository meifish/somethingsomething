#!/bin/bash -       
#title           : batch_fps_conversion.sh
#description     : This script will run ffmpeg on frame extraqction for fps 12
#author		       : Kathryn (https://github.com/meifish)
#date            : 2020-10-03
#usage		       : ./batch_fps_conversion.sh
#==============================================================================

IN_DATA_DIR="./something_videos"
OUT_DATA_DIR="./something_videos_frames"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR};
fi


# The total data size is 220847, too large to read in one time, so batch reading it
declare -i start_video=0
declare -i final_video=220847
declare -i batch=10

num_video=$((final_video-start_video))
END=$((num_video/batch))
echo ${END};

for i in $(seq 0 $END)
do
  if (($i < $END))
  then
    index=$(seq 0 9)
  else
    index=$(seq 0 $((num_video % 10)))
  fi


  for t in ${index[@]}
  do
    video_base="$((start_video+batch*i+t))"
    video_name="${video_base}.webm"
    video_path=$IN_DATA_DIR/$video_name


    indiv_out_dir="${OUT_DATA_DIR}/${video_base/}"
    echo "${video_path}";
    echo "${indiv_out_dir}";

    if [[ ! -d "${indiv_out_dir}" ]]; then
      echo "${indiv_out_dir} doesn't exist. Creating it.";
      mkdir -p ${indiv_out_dir};
    fi

    if [ ! -f "${indiv_out_dir}" ]; then
        ffmpeg -i "${video_path}" -r 12 -q:v 1 "${indiv_out_dir}/%4d.jpg"
    fi

  done
done
