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

#DNA files
BASE_URL="https://ftp.sra.ebi.ac.uk/vol1/fastq/DRR000"
OUTPUT_FILE="${BASEDIR}/dna_data.txt"
TEMP_DIR="${BASEDIR}/temp_dna"

# Create directories
mkdir -p $TEMP_DIR

echo "Starting DNA data extraction from FASTQ files..."

# Function to process a single FASTQ file
process_fastq() {
    local file_path=$1
    local output_file=$2
    
    echo "Processing file: $file_path"
    
    # Extract only the second line of each 4-line record (the raw sequence)
    # Then clean to keep only A, C, G, T characters
    # Append to output without newlines
    awk 'NR % 4 == 2 {
        # Remove all characters that are not A, C, G, or T
        gsub(/[^ACGT]/, "", $0)
        printf "%s", $0
    }' "$file_path" >> "$output_file"
}

# Initialize output file
> $OUTPUT_FILE

# Try to download and process files in the specified range
for i in $(seq -f "%06g" 1 426); do
    PARENT_URL_DIR="DRR${i}"
    # Handle the base case and the _1 and _2 suffixes
    for suffix in "" "_1" "_2"; do
        ID="DRR${i}${suffix}"
        FASTQ_URL="${BASE_URL}/${PARENT_URL_DIR}/${ID}.fastq.gz"
        LOCAL_FILE="${TEMP_DIR}/${ID}.fastq.gz"
        
        echo "Attempting to download: $ID"
        
        # Try to download the file
        wget -q --spider "$FASTQ_URL" 2>/dev/null
        #print wget error code
        EXIT_CODE=$?
        # if exit code is 4, try to download the file again
        while [ $EXIT_CODE -eq 4 ]; do
            wget -q --spider "$FASTQ_URL" 2>/dev/null
            EXIT_CODE=$?
        done
        if [ $EXIT_CODE -eq 0 ]; then
            echo "File exists, downloading: $ID"
            wget -q "$FASTQ_URL" -O "$LOCAL_FILE"
            EXIT_CODE=$?
            # if exit code is 4, try to download the file again
            while [ $EXIT_CODE -eq 4 ]; do
                wget -q --spider "$FASTQ_URL" 2>/dev/null
                EXIT_CODE=$?
            done
            if [ -f "$LOCAL_FILE" ] && [ -s "$LOCAL_FILE" ]; then
                #unzip the file
                gunzip -c "$LOCAL_FILE" > "${LOCAL_FILE%.gz}"
                # Process the downloaded file
                process_fastq "${LOCAL_FILE%.gz}" "$OUTPUT_FILE"
                
                # Remove the downloaded file to save space
                rm "$LOCAL_FILE"
                rm "${LOCAL_FILE%.gz}"
            else
                echo "Failed to download or empty file: $ID"
            fi
        else
            echo "File does not exist: $ID"
        fi
    done
    
    # Print progress every 10 files
    if [ $((i % 10)) -eq 0 ]; then
        echo "Progress: Processed up to file $i of 426"
    fi
done

# Check if output file exists and has content
if [ -f "$OUTPUT_FILE" ] && [ -s "$OUTPUT_FILE" ]; then
    echo "DNA data extraction complete. Output is in $OUTPUT_FILE"
    echo "Final file contains $(wc -c < $OUTPUT_FILE) characters"
else
    echo "Error: No data was extracted or output file is empty"
fi

# Clean up
rmdir $TEMP_DIR 2>/dev/null || echo "Temp directory not empty, some files may remain"

echo "Process completed."

################################################
# Script to download, process and concatenate UniProt database files
# - Downloads uniprot_sprot.dat.gz and uniprot_trembl.dat.gz
# - Extracts only the sequence lines
# - Removes all whitespace
# - Concatenates in order: sprot first, then trembl

# Set variables
BASE_URL="https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete"
OUTPUT_FILE="${BASEDIR}/prot.txt"
TEMP_DIR="${BASEDIR}/temp"

# Create temp directory
mkdir -p $TEMP_DIR

echo "Starting UniProt database processing..."

# Process each database file (sprot first, then trembl)
for DB_ID in "sprot" "trembl"; do
    FILE_NAME="uniprot_${DB_ID}.dat.gz"
    LOCAL_GZ="${TEMP_DIR}/${FILE_NAME}"
    PROCESSED_FILE="${TEMP_DIR}/processed_${DB_ID}.txt"
    
    echo "Processing $FILE_NAME..."
    
    # Download the file
    echo "  Downloading $FILE_NAME..."
    wget -q "${BASE_URL}/${FILE_NAME}" -O "$LOCAL_GZ"
    
    if [ $? -ne 0 ] || [ ! -s "$LOCAL_GZ" ]; then
        echo "  Error downloading $FILE_NAME, skipping..."
        continue
    fi
    
    echo "  Download complete. Extracting sequence lines..."
    
    # Process the file:
    # 1. Decompress gzip
    # 2. Extract only sequence lines (those that don't start with whitespace or '//')
    # 3. Remove all whitespace
    # Note: In UniProt .dat files, sequence lines start with spaces, but the actual
    #       sequence data doesn't start with whitespace
    zcat "$LOCAL_GZ" | awk '
        # We want lines that contain sequence data
        # In UniProt .dat format, sequence lines start with spaces,
        # come after a line with "SQ" and before a line with "//"
        /^SQ/ {in_seq=1; next}
        /^\/\// {in_seq=0; next}
        in_seq && /^     / {
            # This is a sequence line, remove all whitespace
            gsub(/[[:space:]]/,"")
            # Only output non-empty lines
            if (length($0) > 0) {
                print
            }
        }
    ' > "$PROCESSED_FILE"
    
    # Check if processing was successful
    if [ -s "$PROCESSED_FILE" ]; then
        echo "  Successfully processed $FILE_NAME."
    else
        echo "  Error: No sequence data extracted from $FILE_NAME."
    fi
    
    # Remove the downloaded gz file to save space
    rm "$LOCAL_GZ"
    
    # Append to the final output file (creates it if first file)
    cat "$PROCESSED_FILE" >> "$OUTPUT_FILE"
    
    # Remove the processed file
    rm "$PROCESSED_FILE"
    
    echo "  Added $DB_ID sequences to $OUTPUT_FILE"
done

# Check final output
if [ -s "$OUTPUT_FILE" ]; then
    # Remove any remaining newlines to ensure entire output is on a single line
    tr -d '\n' < "$OUTPUT_FILE" > "${OUTPUT_FILE}.tmp" && mv "${OUTPUT_FILE}.tmp" "$OUTPUT_FILE"
    
    echo "Process completed successfully."
    echo "Final output is in $OUTPUT_FILE"
    echo "File contains $(wc -c < $OUTPUT_FILE) characters"
else
    echo "Error: No data was processed or output file is empty"
fi

# Clean up
rmdir $TEMP_DIR 2>/dev/null

echo "UniProt processing complete."