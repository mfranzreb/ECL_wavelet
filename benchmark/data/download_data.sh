#!/bin/bash

BASEDIR=$(dirname "$0")

#!/bin/bash

# Script to download and process Russian Common Crawl data
# - Downloads Russian dataset from the WMT16 repository
# - Decompresses the XZ file
# - Outputs the content to russian_CC.txt

# Set variables
URL="http://web-language-models.s3-website-us-east-1.amazonaws.com/wmt16/deduped/ru.xz"
OUTPUT_FILE="$BASEDIR/russian_CC.txt"
COMPRESSED_FILE="$BASEDIR/ru.xz"

echo "Starting download and processing of Russian language data..."
echo "Source URL: $URL"
echo "Destination file: $OUTPUT_FILE"

# Step 1: Download the XZ file
echo "Downloading compressed file..."
wget -q --show-progress "$URL" -O "$COMPRESSED_FILE"

# Check if download was successful
if [ $? -ne 0 ] || [ ! -s "$COMPRESSED_FILE" ]; then
    echo "Error: Download failed or file is empty."
    exit 1
fi

echo "Download complete. File size: $(du -h $COMPRESSED_FILE | cut -f1)"

# Step 2: Decompress the XZ file directly to the output file
echo "Decompressing file to $OUTPUT_FILE..."
xz -d -c "$COMPRESSED_FILE" > "$OUTPUT_FILE"

# Check if decompression was successful
if [ $? -ne 0 ] || [ ! -s "$OUTPUT_FILE" ]; then
    echo "Error: Decompression failed or output file is empty."
    exit 1
fi

echo "Decompression complete. Output file size: $(du -h $OUTPUT_FILE | cut -f1)"

# Step 3: Clean up the compressed file
echo "Cleaning up temporary files..."
rm "$COMPRESSED_FILE"

echo "Process completed successfully. Russian text data is available in $OUTPUT_FILE"