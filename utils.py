def print_sudoku_grid(tokens):
    """
    Print a 4x4 sudoku grid from model output tokens.
    
    Args:
        tokens (list): List of 16 token IDs representing a 4x4 sudoku grid
    """
    id_to_value_mapping = {
            2: '1',
            3: '2',
            4: '3',
            5: '4',
            6: '.'
        }
    # Convert tokens to sudoku values
    grid_values = []
    for token in tokens:
        value = id_to_value_mapping[token]
        grid_values.append(value)
    
    # Print the grid with borders
    print("+---+---+---+---+")
    for i in range(4):
        row_start = i * 4
        row_values = grid_values[row_start:row_start + 4]
        print(f"| {' | '.join(row_values)} |")
        print("+---+---+---+---+")