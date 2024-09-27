#!/bin/bash

# Check if a file path is provided
if [ $# -eq 0 ]; then
  echo "Please provide the path to an MP4 file."
  exit 1
fi

# Get the file path from the command line argument
file_path="$1"

# Check if the file exists
if [ ! -f "$file_path" ]; then
  echo "File not found: $file_path"
  exit 1
fi

# Extract the frame rate using FFmpeg
frame_rate=$(ffmpeg -i "$file_path" 2>&1 | grep -oP "fps,.*tb\(r\).*" | grep -oP "\d+(?:\.\d+)? fps")

# Check if frame rate was successfully extracted
if [ -z "$frame_rate" ]; then
  echo "Unable to extract frame rate from the file."
  exit 1
fi

# Echo the frame rate
echo "Frame rate of $file_path: $frame_rate"
