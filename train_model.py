import os
import argparse
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# Установить устройство для использования CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_dataset = load_from_disk('train_dataset')
test_dataset = load_from_disk('test_dataset')


def check_labels(dataset):
    for example in dataset:
        label = example['labels']
        if label < 0 or label >= 28:
            print(f"Invalid label found: {label}")
            return False
    return True


if not check_labels(train_dataset) or not check_labels(test_dataset):
    print("Dataset contains invalid labels. Exiting...")
    exit()


# Обработка аргументов командной строки
parser = argparse.ArgumentParser(description="Train BERT model")
parser.add_argument('--checkpoint', type=str, help="Path to checkpoint directory")
args = parser.parse_args()


if args.checkpoint:
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = BertForSequenceClassification.from_pretrained(args.checkpoint)
else:
    print("Training model from scratch")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-multilingual-cased',
        num_labels=28
    )

model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',  # directory for storing logs
    logging_steps=1000,
    save_steps=10000
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Продолжить обучение, если указан чекпоинт
if args.checkpoint:
    trainer.train(resume_from_checkpoint=args.checkpoint)
else:
    trainer.train()


model.save_pretrained('./model')
print("Модель успешно дообучена и сохранена")
