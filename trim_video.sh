# Usage: ./trim_video.sh vid 88
# check two variables, if not, print usage
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: $0 vid frames_to_have"
  exit 1
fi

output_file="${1%.*}_trimmed.mp4"

ffmpeg -i $1 -vf "select=lte(n\,$2-1)" -vsync vfr $output_file
