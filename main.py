from dataset import generate_one_sample
from transformers import T5ForConditionalGeneration
from utils import print_sudoku_grid
import torch

model = T5ForConditionalGeneration.from_pretrained("Somsung/t5-sudoku-solver")

#Test the model
input_ids, labels = generate_one_sample(n_clues=8)
input_tensor = torch.tensor([input_ids]).to(model.device)  # Add batch dimension

print_sudoku_grid(input_ids[:-1])
print_sudoku_grid(labels[:-1])

output = model.generate(input_tensor)
print_sudoku_grid(output[0].tolist()[1:-1])