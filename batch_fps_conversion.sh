IN_DATA_DIR="./something_videos"
OUT_DATA_DIR="./something_videos_frames"

if [[ ! -d "${OUT_DATA_DIR}" ]]; then
  echo "${OUT_DATA_DIR} doesn't exist. Creating it.";
  mkdir -p ${OUT_DATA_DIR};
fi


# The total data size is 220847, too large to read in one time, so batch reading it
declare -i total_video=220847
declare -i batch=10
END=$((total_video/batch))
echo ${END};

for i in $(seq 0 $END)
do
  if (($i < $END))
  then
    index=$(seq 1 10)
  else
    index=$(seq 1 $((total_video % 10)))
  fi


  for t in ${index[@]}
  do
    video_base="$((batch*i+t))"
    video_name="${video_base}.webm"
    video_path=$IN_DATA_DIR/$video_name
    echo "${video_path}";

    indiv_out_dir="${OUT_DATA_DIR}/${video_base/}"
    mkdir -p ${indiv_out_dir}

    if [ ! -f "${indiv_out_dir}" ]; then
        ffmpeg -i "${video_path}" -r 12 -q:v 1 "${indiv_out_dir}/%4d.jpg"
    fi

  done
done
