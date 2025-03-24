import numpy as np
import random

def board_to_bitboard(board):
    # 64 bits, ie 1100.. where 1 corresponds to top left pos on board
    bitboard = 0
    for i in range(8):
        for j in range(8):
            # j should correspond to x (column, second value)
            bitboard = (bitboard << 1) | int(board[i][j])
    return bitboard

def bitboard_to_board(bitboard):
    board = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            # column (x, second val) changes more frequently
            # j changes more frequently
            next_bit = bitboard % 2
            bitboard = bitboard >> 1
            board[7-i][7-j] = next_bit
    return board

def clear_position(position):
    # clear lines
    clear_rows = []
    clear_cols = []

    new_position = position

    row_mask = 0xFF00000000000000
    col_mask = 0x8080808080808080
    full_mask = 0xFFFFFFFFFFFFFFFF
    
    for i in range(8):
        row_mask_shifted = row_mask >> (8 * i)
        if (position & row_mask_shifted == row_mask_shifted):
            clear_rows.append(i)
    
    for j in range(8):
        col_mask_shifted = col_mask >> j
        if (position & col_mask_shifted == col_mask_shifted):
            clear_cols.append(j)
    
    for row in clear_rows:
        new_position &= (~(row_mask >> (8 * row)) & full_mask)
    
    for col in clear_cols:
        new_position &= (~(col_mask >> col) & full_mask)
    
    return new_position

def clear_position_info(position):
    # clear lines
    clear_rows = []
    clear_cols = []

    new_position = position
    clear_count = 0

    row_mask = 0xFF00000000000000
    col_mask = 0x8080808080808080
    full_mask = 0xFFFFFFFFFFFFFFFF
    
    for i in range(8):
        row_mask_shifted = row_mask >> (8 * i)
        if (position & row_mask_shifted == row_mask_shifted):
            clear_rows.append(i)
    
    for j in range(8):
        col_mask_shifted = col_mask >> j
        if (position & col_mask_shifted == col_mask_shifted):
            clear_cols.append(j)
    
    for row in clear_rows:
        clear_count += 1
        new_position &= (~(row_mask >> (8 * row)) & full_mask)
    
    for col in clear_cols:
        clear_count += 1
        new_position &= (~(col_mask >> col) & full_mask)
    
    return new_position, clear_count

def get_resulting_position(current_position, block_info, placement):
    block_bitboard = block_info[0]
    block = block_bitboard >> (8 * placement[1] + placement[0])

    return current_position | block

def generate_positions_end_clear(current_position, block_info, positions_clear, positions_nonclear, permutations, placements):
    block_bitboard, block_width, block_height = block_info

    for i in range(8-block_width+1):
        for j in range(8-block_height+1):
            # Shift right by i, shift down by j
            block = block_bitboard >> (8 * j + i)
            if (block & current_position == 0):
                position, cc = clear_position_info(block | current_position)

                if cc > 0 and position not in positions_clear:
                    full_placement = placements + [[i, j]]
                    positions_clear[position] = (permutations, full_placement)
                
                if cc == 0 and position not in positions_nonclear:
                    full_placement = placements + [[i, j]]
                    positions_nonclear[position] = (permutations, full_placement)

def generate_positions_clear(current_position, blocks_labeled, positions_clear, positions_nonclear, permutations=[], placements=[]):
    if len(blocks_labeled) == 1:
        label, block = blocks_labeled[0]
        full_permutation = permutations + [label]
        generate_positions_end_clear(current_position, block, positions_clear, positions_nonclear, full_permutation, placements)
    else:
        for index, (label, block) in enumerate(blocks_labeled):
            full_permutation = permutations + [label]
            block_bitboard, block_width, block_height = block

            for i in range(8-block_width+1):
                for j in range(8-block_height+1):
                    # Shift right by i, shift down by j
                    block = block_bitboard >> (8 * j + i)

                    if (block & current_position == 0):
                        new_position = clear_position(block | current_position)

                        full_placement = placements + [[i, j]]
                        new_blocks = blocks_labeled[:index] + blocks_labeled[index+1:]
                        
                        generate_positions_clear(new_position, new_blocks, positions_clear, positions_nonclear, full_permutation, full_placement)

def generate_positions_end(current_position, block_info, positions, permutations, placements):
    block_bitboard, block_width, block_height = block_info

    for i in range(8-block_width+1):
        for j in range(8-block_height+1):
            # Shift right by i, shift down by j
            block = block_bitboard >> (8 * j + i)
            if (block & current_position == 0):
                position = clear_position(block | current_position)

                if position not in positions:
                    full_placement = placements + [[i, j]]
                    positions[position] = (permutations, full_placement)

def generate_positions(current_position, blocks_labeled, positions, permutations=[], placements=[]):
    if len(blocks_labeled) == 1:
        label, block = blocks_labeled[0]
        full_permutation = permutations + [label]
        generate_positions_end(current_position, block, positions, full_permutation, placements)
    else:
        for index, (label, block) in enumerate(blocks_labeled):
            full_permutation = permutations + [label]
            block_bitboard, block_width, block_height = block

            for i in range(8-block_width+1):
                for j in range(8-block_height+1):
                    # Shift right by i, shift down by j
                    block = block_bitboard >> (8 * j + i)

                    if (block & current_position == 0):
                        new_position = clear_position(block | current_position)

                        full_placement = placements + [[i, j]]
                        new_blocks = blocks_labeled[:index] + blocks_labeled[index+1:]
                        
                        generate_positions(new_position, new_blocks, positions, full_permutation, full_placement)

def get_bitboard_sum(bitboard):
    return bitboard.bit_count()

def block_id_to_block_info(block_id):
    # this function doesn't have to be fast
    block = np.zeros((8, 8), dtype=int)
    max_x = 0
    max_y = 0
    for j in range(5):
        for i in range(5):
            if (block_id % 2 == 1):
                max_x = max(max_x, i)
                max_y = max(max_y, j)
                block[j, i] = 1
            block_id = block_id >> 1
    pos = np.zeros((8, 8))
    pos[0][0] = 1

    return board_to_bitboard(block), max_x+1, max_y+1

def print_position(position):
    # ANSI color codes for different colors
    colors = [
        '\033[92m',  # Green
    ]
    reset = '\033[0m'  # Reset color
    
    # Choose a random color for the entire position
    color = random.choice(colors)
    
    # Print the 8x8 position with color
    for row in position:
        print(''.join(f'{color}██{reset}' if cell == 1 else '⋅⋅' for cell in row))
    print()

if __name__ == "__main__":
    pos = np.zeros((8, 8))

    for i in range(8):
        for j in range(8):
            if ((i + j) % 2 == 0 or j == 3 or i == 2):
                pos[i][j] = 1

    bitboard = board_to_bitboard(pos)
    bitboard = clear_position(bitboard)

    positions = {}
    block_bitboard, block_width, block_height = block_id_to_block_info(1122)
    print(block_width, block_height)
    # print(bitboard_to_board(block_bitboard))

    generate_positions(bitboard, [(0, block_id_to_block_info(1122)), (1, block_id_to_block_info(71))], positions)
    print(len(positions))

    for position, (permutation, placement) in positions.items():
        print(permutation, placement)
        print_position(bitboard_to_board(position))

    # print_position(bitboard_to_board(bitboard))
    print(get_bitboard_sum(bitboard))