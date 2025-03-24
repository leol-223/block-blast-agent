import pickle
import numpy as np
import random
import matplotlib.pyplot as plt

def print_block_from_id(id):
    block = np.zeros((5, 5), dtype=int)
    for j in range(5):
        for i in range(5):
            if (id % 2 == 1):
                block[j, i] = 1
            id = id // 2
            
    # ANSI color codes for different colors
    colors = [
        '\033[91m',  # Red
        '\033[92m',  # Green
        '\033[93m',  # Yellow
        '\033[94m',  # Blue
        '\033[95m',  # Purple
    ]
    reset = '\033[0m'  # Reset color
    
    # Choose a random color for this block
    color = random.choice(colors)
    
    # Print the block with color
    for row in block:
        print(''.join(f'{color}██{reset}' if cell == 1 else '⋅⋅' for cell in row))
    print()  # Add an extra newline for better spacing

def print_blocks_from_ids(*ids):
    # Create blocks
    blocks = []
    for id in ids:
        block = np.zeros((5, 5), dtype=int)
        for j in range(5):
            for i in range(5):
                if (id % 2 == 1):
                    block[j, i] = 1
                id = id // 2
        blocks.append(block)
    
    # ANSI color codes for different colors
    colors = [
        '\033[91m',  # Red
        '\033[92m',  # Green
        '\033[93m',  # Yellow
        '\033[94m',  # Blue
        '\033[95m',  # Purple
    ]
    reset = '\033[0m'  # Reset color
    
    # Assign a random color to each block
    block_colors = [random.choice(colors) for _ in blocks]
    
    # Print all blocks row by row
    for row_idx in range(5):
        # For each block, print the current row
        for block, color in zip(blocks, block_colors):
            row = block[row_idx]
            print(''.join(f'{color}██{reset}' if cell == 1 else '⋅⋅' for cell in row), end=' ')
        print()  # New line after each row
    print()  # Extra newline for spacing

blocks_encountered = []
# Open the file in binary read mode ('rb')
with open('blocks_encountered_new.pkl', 'rb') as f:
    blocks_encountered = pickle.load(f)

total_blocks = 0
id_counts = {}
id_counts_desparate = {}

total_blocks_difficult = 0

bitboard_num_1s = []

scatter_counts = []

for bitboard, (block_left, block_middle, block_right) in blocks_encountered:
    count = bitboard.bit_count()
    count_blocks = block_left.bit_count() + block_middle.bit_count() + block_right.bit_count()
    scatter_counts.append([count, count_blocks])
    bitboard_num_1s.append(count)
    if count < 35:
        if block_left in id_counts:
            id_counts[block_left] += 1
        else:
            id_counts[block_left] = 1
        if block_middle in id_counts:
            id_counts[block_middle] += 1
        else:
            id_counts[block_middle] = 1
        if block_right in id_counts:
            id_counts[block_right] += 1
        else:
            id_counts[block_right] = 1
    else:
        if block_left in id_counts_desparate:
            id_counts_desparate[block_left] += 1
        else:
            id_counts_desparate[block_left] = 1
        if block_middle in id_counts_desparate:
            id_counts_desparate[block_middle] += 1
        else:
            id_counts_desparate[block_middle] = 1
        if block_right in id_counts_desparate:
            id_counts_desparate[block_right] += 1
        else:
            id_counts_desparate[block_right] = 1
        total_blocks_difficult += 3
    total_blocks += 3

sorted_id_counts = dict(sorted(id_counts.items(), key=lambda x: x[1], reverse=True))
print("Sorted counts:")
for block_id, count in sorted_id_counts.items():
    print(f"Frequency (Id {block_id}): \033[91m{round(count/(total_blocks-total_blocks_difficult)*100, 2)}%\033[0m")
    print_block_from_id(block_id)
    print()

    sorted_id_counts = dict(sorted(id_counts.items(), key=lambda x: x[1], reverse=True))

sorted_id_counts_d = dict(sorted(id_counts_desparate.items(), key=lambda x: x[1], reverse=True))
print("Sorted counts (difficult positions):")
for block_id, count in sorted_id_counts_d.items():
    print(f"Frequency (Id {block_id}): \033[91m{round(count/total_blocks_difficult*100, 2)}%\033[0m")
    print_block_from_id(block_id)
    print()

total_triples = 0
id_counts = {}
for bitboard, (block_left, block_middle, block_right) in blocks_encountered:
    block_triple = tuple(sorted([block_left, block_middle, block_right]))
    if block_triple in id_counts:
        id_counts[block_triple] += 1
    else:
        id_counts[block_triple] = 1
    total_triples += 1

sorted_id_counts = dict(sorted(id_counts.items(), key=lambda x: x[1], reverse=True))
print(f"Sorted counts ({total_triples} total):")
for triple, count in sorted_id_counts.items():
    if count < 5:
        continue
    triple_copy = list(triple)
    random.shuffle(triple_copy)
    left_id, middle_id, right_id = triple_copy
    print(f"Frequency (Ids {left_id}, {middle_id}, {right_id}): \033[91m{round(count/total_triples*100, 2)}%\033[0m")
    print_blocks_from_ids(left_id, middle_id, right_id)
    print()

# Create a figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# First plot (histogram)
ax1.hist(bitboard_num_1s, bins=range(57), align='left', rwidth=0.8)
ax1.set_title('Distribution of Block Count')
ax1.set_xlabel('Number of 1s')
ax1.set_ylabel('Frequency')
ax1.grid(True, alpha=0.3)

# Second plot (scatter)
x = [point[0] for point in scatter_counts]
y = [point[1] for point in scatter_counts]
ax2.scatter(x, y, alpha=0.5)
ax2.set_title('Scatter Plot')
ax2.set_xlabel('Block counts (board)')
ax2.set_ylabel('Block counts (blocks given)')
ax2.grid(True, alpha=0.3)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()