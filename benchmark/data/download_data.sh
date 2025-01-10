#!/bin/bash

DIR=$(dirname $0)

#CommonCrawl files
CC="https://data.commoncrawl.org/crawl-data/CC-MAIN-2019-09/segments/1550247479101.30/wet/CC-MAIN-20190215183319-20190215205319-#ID.warc.wet.gz"
CC_out="$DIR/common_crawl.txt"

#Download all CC files with "#ID" from "00000" to "00600" and append them to CC_out
for i in $(seq -w 0 600); do
    FILE_URL="${CC//#ID/00$i}"
    curl -s "$FILE_URL" >> "$DIR/tmp.warc.wet.gz"
    gunzip -c "$DIR/tmp.warc.wet.gz" >> "$DIR/tmp.warc.wet"
    # Remove lines that contain "WARC/1.0" and the following 8 lines
    sed -i '/WARC\/1.0/,+8d' "$DIR/tmp.warc.wet"

    #Remove html tags
    sed -e 's/<[^>]*>//g' "$DIR/tmp.warc.wet" >> "$CC_out"
done

#DNA files
