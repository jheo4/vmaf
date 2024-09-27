# ./run.sh vid1.mp4 vid2.mp4
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <video1> <video2>"
  exit 1
fi

first_vid=$(basename $1)
second_vid=$(basename $2)

echo "$first_vid, $second_vid"

docker run --rm -v $(pwd)/vid_temp:/files gfdavila/easyvmaf \
  -r /files/$first_vid -d /files/$second_vid
