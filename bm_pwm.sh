#!/bin/bash

data_size=8000000000
alphabet_sizes=(4 6 8 12 16 24 32 48 64 96 128 192 256 384 512 768 1024 1536 2048 3072 4096 6144 8192 12288 16384 24576 32768 49152 65536)

# build data generator
g++ -O3 -fopenmp -o data_gen generate_random_data.cpp

echo "algorithm,data_size,alphabet_size,median_time(ms)" > pwm_random.csv

file_path="data.txt"

for alphabet_size in ${alphabet_sizes[@]}; do
    echo "Alphabet size: $alphabet_size"
    ./data_gen $data_size $alphabet_size $file_path
    #var char_size is 1 if alphabet_size <=256 else 2
    char_size=1
    if [ $alphabet_size -gt 256 ]; then
        char_size=2
    fi
    ./third_party/pwm/build/src/benchmark -f ./data.txt -b $char_size -r 10 -l $data_size --parallel --no_huffman --no_matrices | \
    while IFS= read -r line; do
        # Check if line starts with "RESULT"
        if [[ "$line" =~ ^RESULT ]]; then
            # Extract the required fields using grep and sed
            algo=$(echo "$line" | grep -o "algo=[^ ]*" | sed 's/algo=//')
            characters=$(echo "$line" | grep -o "characters=[^ ]*" | sed 's/characters=//')
            sigma=$(echo "$line" | grep -o "sigma=[^ ]*" | sed 's/sigma=//')
            median_time=$(echo "$line" | grep -o "median_time=[^ ]*" | sed 's/median_time=//')
            
            # Append to output file
            if [[ -n "$algo" && -n "$characters" && -n "$sigma" && -n "$median_time" ]]; then
                echo "$algo,$characters,$sigma,$median_time" >> pwm_random.csv
            fi
        fi
    done
done

