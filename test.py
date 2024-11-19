import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, required=True)
    args = parser.parse_args()
    num_warps = args.n

    best_match = None  # Track the best option found
    best_difference = float("inf")  # Initialize with infinity for comparison

    max_block_size = 16  # Maximum block size supported by the GPU
    for k in range(1, max_block_size + 1):
        # Check if max_block_size is divisible by k
        if max_block_size % k != 0:
            continue

        # Compute block_size and num_blocks
        block_size = max_block_size // k
        if block_size < 4:
            break
        num_blocks = num_warps / block_size

        # Check if num_blocks is an integer
        if num_blocks.is_integer():
            print("Perfect match found!")
            print(f"Block size: {block_size}, Num blocks: {int(num_blocks)}")
            break

        # Otherwise, track the closest match
        product = block_size * round(num_blocks)  # Approximate the nearest product
        difference = num_warps - product
        # only save difference if product is less than num_warps
        if difference > 0 and difference < best_difference:
            best_difference = difference
            best_match = (block_size, round(num_blocks))

    print("Best match found!")
    print(f"Block size: {best_match[0]}, Num blocks: {best_match[1]}")
