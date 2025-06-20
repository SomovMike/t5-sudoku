import random
import copy
from datasets import Dataset

SIZE = 4
BLOCK = 2
NUMBERS = list(range(1, SIZE + 1))

def is_valid(board, row, col, num):
    for i in range(SIZE):
        if board[row][i] == num or board[i][col] == num:
            return False
    start_row, start_col = row - row % BLOCK, col - col % BLOCK
    for i in range(BLOCK):
        for j in range(BLOCK):
            if board[start_row + i][start_col + j] == num:
                return False
    return True

def find_empty(board):
    for i in range(SIZE):
        for j in range(SIZE):
            if board[i][j] == 0:
                return i, j
    return None

def solve(board, count_solutions=False):
    empty = find_empty(board)
    if not empty:
        return 1 if count_solutions else True
    row, col = empty
    num_solutions = 0
    for num in NUMBERS:
        if is_valid(board, row, col, num):
            board[row][col] = num
            result = solve(board, count_solutions)
            board[row][col] = 0
            if count_solutions:
                num_solutions += result
                if num_solutions > 1:
                    break
            else:
                return True
    return num_solutions if count_solutions else False

def generate_full_board():
    board = [[0]*SIZE for _ in range(SIZE)]
    def fill():
        empty = find_empty(board)
        if not empty:
            return True
        row, col = empty
        random.shuffle(NUMBERS)
        for num in NUMBERS:
            if is_valid(board, row, col, num):
                board[row][col] = num
                if fill():
                    return True
                board[row][col] = 0
        return False
    fill()
    return board

def generate_sudoku(n_clues=8):
    solution = generate_full_board()
    puzzle = copy.deepcopy(solution)
    cells = [(i, j) for i in range(SIZE) for j in range(SIZE)]
    random.shuffle(cells)

    for row, col in cells:
        if sum(puzzle[i][j] != 0 for i in range(SIZE) for j in range(SIZE)) <= n_clues:
            break
        temp = puzzle[row][col]
        puzzle[row][col] = 0
        test_board = copy.deepcopy(puzzle)
        if solve(test_board, count_solutions=True) != 1:
            puzzle[row][col] = temp  # revert if multiple solutions

    return puzzle, solution


def convert_grid_to_tokens(grid):
    result = []
    for row in grid:
        for num in row:
            result.append('empty' if num == 0 else str(num))

    token_mapping = {"1": 2, "2": 3, "3": 4, "4": 5, "empty": 6}
    result = [token_mapping[token] for token in result] + [1] #EOS token
    return result


def generate_tokenized_dataset(n_samples=1000, min_clues=8, max_clues=13):
    input_ids = []
    labels = []
    for _ in range(n_samples):    
        puzzle, solution = generate_sudoku(n_clues=random.randint(min_clues, max_clues))
        input_ids.append(convert_grid_to_tokens(puzzle))
        labels.append(convert_grid_to_tokens(solution))


    dataset_dict = {
        "input_ids": input_ids,
        "labels": labels
    }
    tokenized_dataset = Dataset.from_dict(dataset_dict)
    return tokenized_dataset


def generate_one_sample(n_clues=8):
    puzzle, solution = generate_sudoku(n_clues=n_clues)
    input_ids = convert_grid_to_tokens(puzzle)
    labels = convert_grid_to_tokens(solution)
    return input_ids, labels