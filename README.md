# T5 Sudoku Solver 🧩

This project demonstrates how to adapt Google's T5 (Text-to-Text Transfer Transformer) for solving 4x4 Sudoku puzzles, showcasing the versatility of encoder-decoder architectures beyond traditional NLP tasks. By treating Sudoku as a sequence-to-sequence problem, the model learns to map incomplete grids to their complete solutions through transformer attention mechanisms.

🤗 **Pre-trained model available**: [Somsung/t5-sudoku-solver](https://huggingface.co/Somsung/t5-sudoku-solver)

## 🏗️ Architecture Overview

```
Input Sudoku Grid → T5 Encoder → Latent Representation → T5 Decoder → Solution Grid

[2, 0, 4, 3]     →   Encoder   →   Hidden States   →   Decoder   →   [2, 1, 4, 3]
[4, 3, 0, 0]                                                        [4, 3, 2, 1]  
[0, 4, 0, 2]                                                        [3, 4, 1, 2]
[0, 0, 3, 4]                                                        [1, 2, 3, 4]
```

The encoder processes the incomplete puzzle as a sequence, decoder generates the solution autoregressively.

## 🔧 Setup

```bash
pip install -r requirements.txt
python main.py
```

## 🧠 Technical Stuff

**Custom Tokenization**: I used a custom vocabulary with just 7 tokens instead of 32000:
```python
"<pad>": 0, "</s>": 1, "1": 2, "2": 3, "3": 4, "4": 5, "empty": 6
```

**No Dropout**: Disabled dropout since Sudoku is deterministic - every token matters for logical reasoning.

**Grid as Sequence**: Flattened 4x4 grid into 16-token sequence + EOS token for seq2seq training.

**Custom Embeddings**: Instead of learning new embeddings from scratch, I copied existing T5 embeddings for numbers "1"-"4" and word "empty" as initialization for my custom tokens, then resized T5's vocabulary.

## 📁 Files

- `T5SudokuModel.py` - Custom T5 wrapper with Sudoku tokenization
- `dataset.py` - Generates puzzles and handles data preprocessing  
- `train.py` - Training pipeline
- `main.py` - Run inference and print actual grids

Trained on 100k generated puzzles (with different difficulty) with T5-small (60M params). 
Results of evaluation could be seen in [eval_model.ipynb](eval_model.ipynb) notebook.