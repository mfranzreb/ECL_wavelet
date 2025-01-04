import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tune_file", type=str)
    parser.add_argument("GPU_name", type=str)
    args = parser.parse_args()

    # Check if any of the GPU names are already in the tune file
    with open(args.tune_file, "r") as f:
        file_string = f.read()
        if args.GPU_name in file_string:
            sys.exit(1)

    sys.exit(0)
