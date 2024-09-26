docker run --rm -v $(pwd):/files vmaf \
  -p 420 -w 718 -h 508 -b 8 \
  -r /files/t1.yuv \
  -d /files/t2.yuv --output /files/result.txt
