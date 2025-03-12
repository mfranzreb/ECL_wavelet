#!/bin/bash

BASEDIR=$(dirname "$0")

# Script to process CommonCrawl WET files
# - Downloads WET files with IDs from 00000 to 00600
# - Removes WARC metadata (WARC/1.0 line and following 8 lines)
# - Concatenates files in ascending order by ID
# Set variables
BASE_URL="https://data.commoncrawl.org/crawl-data/CC-MAIN-2019-09/segments/1550247479101.30/wet"
FILE_PREFIX="CC-MAIN-20190215183319-20190215205319-"
FILE_SUFFIX=".warc.wet.gz"
OUTPUT_DIR="common_crawl"
DOWNLOAD_FOLDER="${BASEDIR}/${OUTPUT_DIR}"
FINAL_OUTPUT="${BASEDIR}/common_crawl.txt"



echo " Download folder: $DOWNLOAD_FOLDER"

# Create directories
mkdir -p "$DOWNLOAD_FOLDER"
echo "Starting WET file processing..."

# Download and process each file
for i in $(seq -f "%05g" 0 2); do
  FILENAME="${FILE_PREFIX}${i}${FILE_SUFFIX}"
  DOWNLOAD_PATH="${DOWNLOAD_FOLDER}/${FILENAME}"
  PROCESSED_PATH="${DOWNLOAD_FOLDER}/processed_${i}.txt"
  
  echo "Processing file $i of 2: $FILENAME"
  
  # Download the file if it doesnt exist
  if [ ! -f "$DOWNLOAD_PATH" ]; then
    echo "  Downloading $FILENAME..."
    wget -q "${BASE_URL}/${FILENAME}" -O "$DOWNLOAD_PATH"
    
    # Check if download was successful
    if [ $? -ne 0 ]; then
      echo "  Error downloading $FILENAME, skipping..."
      continue
    fi
  else
    echo "  $FILENAME already exists, skipping download..."
  fi

# Uncompress the file
    echo "  Uncompressing $FILENAME..."
    gunzip -c "$DOWNLOAD_PATH" > "${DOWNLOAD_PATH%.gz}"
  
  # Process the file - remove WARC metadata (line with WARC/1.0 and the following 8 lines)
  echo "  Removing WARC metadata..."
  awk '
    /^WARC\/1.0/ {
      # Skip this line and the next 8 lines
      for(i=0; i<9; i++) {
        getline
      }
      next
    }
    { print }
  ' "${DOWNLOAD_PATH%.gz}" > "$PROCESSED_PATH"
  
  echo "  Done processing $FILENAME"

  # remove uncompressed and unprocessed file
    rm "${DOWNLOAD_PATH%.gz}"
    rm "$DOWNLOAD_PATH"
done

echo "All files processed. Concatenating in ascending order..."

# Concatenate all processed files in order
for i in $(seq -f "%05g" 0 2); do
  PROCESSED_PATH="${DOWNLOAD_FOLDER}/processed_${i}.txt"
  
  # Check if processed file exists before concatenating
  if [ -f "$PROCESSED_PATH" ]; then
    cat "$PROCESSED_PATH" >> "$FINAL_OUTPUT"
    echo "Added $PROCESSED_PATH to $FINAL_OUTPUT"
  fi
  # remove processed file
    rm "$PROCESSED_PATH"
done

echo "Completed! Final output is in $FINAL_OUTPUT"