# Usage: ./trim_video.sh vid 88
# check two variables, if not, print usage
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 vid frames_to_have"
  exit 1
fi

ffmpeg -i shorter_video.mp4 -vf "tpad=stop_mode=clone:stop_frame=$2" $1
