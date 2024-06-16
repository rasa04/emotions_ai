from datetime import time

import torch
from datasets import load_from_disk
from rich.console import Console
from rich.table import Table

from model import predict_emotions, emotions

# Initialize rich console
console = Console()

DATA_COUNT = 100

# Load datasets
test_dataset = load_from_disk('test_dataset')  # Assuming test_dataset is saved

# Convert to pandas DataFrame for easier access
df_test = test_dataset.to_pandas().sample(n=DATA_COUNT, random_state=time().microsecond % 100)

# Load the model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_model(df_test_ds, text_column, label_column):
    correct_predictions = 0
    correct_top5_predictions = 0
    total_predictions = len(df_test_ds)

    table = Table(title="Emotion Prediction Evaluation")
    if DATA_COUNT <= 100:
        table.add_column("Text", justify="left", style="cyan", no_wrap=True)
        table.add_column("Expected Emotion", justify="center", style="magenta")
        table.add_column("Predicted Emotion", justify="center", style="green")
        table.add_column("Result", justify="center")
        table.add_column("Top 5 Contains Correct", justify="center", style="yellow")

    for idx, example in df_test_ds.iterrows():
        text = example[text_column][:64]
        top_emotions, input_tokens, output_tokens = predict_emotions(text)
        top5_emotions = [item['emotion'] for item in top_emotions]
        predicted_emotion = top5_emotions[0]
        expected_emotion = emotions[example[label_column]]
        is_correct = predicted_emotion == expected_emotion
        is_correct_top5 = expected_emotion in top5_emotions

        if DATA_COUNT <= 100:
            result_color = "green" if is_correct else "red"
            table.add_row(
                str(text),
                str(expected_emotion),
                str(predicted_emotion),
                f"[{result_color}]{'Correct' if is_correct else 'Incorrect'}[/{result_color}]",
                "Yes" if is_correct_top5 else "No"
            )

        if is_correct:
            correct_predictions += 1
        if is_correct_top5:
            correct_top5_predictions += 1

    accuracy = correct_predictions / total_predictions * 100
    top5_accuracy = correct_top5_predictions / total_predictions * 100
    console.print(table)
    console.print(f'\nAccuracy (Top 1): {accuracy:.2f}%\n', style="bold yellow")
    console.print(f'Accuracy (Top 5): {top5_accuracy:.2f}%\n', style="bold yellow")


console.print("Evaluating model on Russian test messages:", style="bold blue")
evaluate_model(df_test, text_column='ru_text', label_column='labels')

console.print("Evaluating model on English test messages:", style="bold blue")
evaluate_model(df_test, text_column='text', label_column='labels')
