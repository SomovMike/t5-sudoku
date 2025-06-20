from transformers import TrainingArguments, Trainer
from T5SudokuModel import T5SudokuModel
from dataset import generate_tokenized_dataset
from transformers import TrainingArguments, Trainer

model = T5SudokuModel()
tokenized_dataset = generate_tokenized_dataset(n_samples=100000, min_clues=8, max_clues=10)
train_test_split = tokenized_dataset.train_test_split(test_size=0.01, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']



training_args = TrainingArguments(
    output_dir="./t5-sudoku-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-4,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir="./logs",
    save_steps=600,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=3,
    dataloader_drop_last=True,
    report_to=[],
    fp16=True,
)

trainer = Trainer(
    model=model.model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()