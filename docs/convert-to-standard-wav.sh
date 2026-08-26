#!/bin/bash

# Convert to Standard WAV
# See docs/convert-to-standard-wav.md for full documentation.

# Check if the input file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <input_wav_file>"
  exit 1
fi

# Input file
input_file="$1"

# Output file name
output_file="standard_${input_file}"

# Run the ffmpeg command
ffmpeg -i "$input_file" -acodec pcm_s16le -ar 22050 -ac 1 "$output_file"

# Check if the conversion was successful
if [ $? -eq 0 ]; then
  echo "Conversion successful! Output file: $output_file"
else
  echo "Conversion failed."
fi
