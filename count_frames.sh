# Usage: ./count_frames.sh video.mp4
#
if [ -z "$1" ]; then
  echo "Usage: $0 video.mp4"
  exit 1
fi

ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 $1
