import requests
from rich.console import Console
from rich.table import Table

# Initialize rich console
console = Console()

# Enhanced and more complex test messages
test_messages_ru = [
    {"text": "Сегодня невероятный день! Спасибо за превосходное обслуживание.", "expected_emotion": "joy"},
    {"text": "Ваш продукт оставляет желать лучшего, я крайне недоволен.", "expected_emotion": "disapproval"},
    {"text": "Ваш сервис разочаровывает меня снова и снова.", "expected_emotion": "disappointment"},
    {"text": "Ваш магазин - лучший в городе, буду рекомендовать всем знакомым.", "expected_emotion": "admiration"},
    {"text": "Ваша политика возврата вызывает у меня тревогу.", "expected_emotion": "fear"},
    {"text": "Я не уверен, стоит ли продолжать покупки у вас.", "expected_emotion": "confusion"},
    {"text": "Обслуживание на высоте, я более чем доволен.", "expected_emotion": "gratitude"},
    {"text": "Меня раздражает, что доставка занимает так много времени.", "expected_emotion": "annoyance"},
    {"text": "Я опечален, так как мой заказ потерялся.", "expected_emotion": "sadness"},
    {"text": "Большое спасибо за вашу помощь, это было очень полезно.", "expected_emotion": "gratitude"},
    {"text": "Ваши действия приносят мне огромное удовольствие.", "expected_emotion": "joy"},
    {"text": "Качество обслуживания оставляет желать лучшего.", "expected_emotion": "disapproval"},
    {"text": "Я чрезвычайно разочарован вашим сервисом.", "expected_emotion": "disappointment"},
    {"text": "Вы заслуживаете самых высоких похвал.", "expected_emotion": "admiration"},
    {"text": "Меня ужасает ваша политика обмена товаров.", "expected_emotion": "fear"},
    {"text": "Я в замешательстве от ваших условий возврата.", "expected_emotion": "confusion"},
    {"text": "Ваш сервис просто великолепен, благодарю.", "expected_emotion": "gratitude"},
    {"text": "Меня выводит из себя задержка с доставкой.", "expected_emotion": "annoyance"},
    {"text": "Я глубоко расстроен потерей моего заказа.", "expected_emotion": "sadness"},
    {"text": "Спасибо за поддержку, это действительно помогло.", "expected_emotion": "gratitude"}
]

test_messages_en = [
    {"text": "Today is an incredible day! Thank you for the excellent service.", "expected_emotion": "joy"},
    {"text": "Your product leaves much to be desired, I am extremely dissatisfied.", "expected_emotion": "disapproval"},
    {"text": "Your service disappoints me time and again.", "expected_emotion": "disappointment"},
    {"text": "Your store is the best in town, I will recommend it to everyone I know.",
     "expected_emotion": "admiration"},
    {"text": "Your return policy makes me anxious.", "expected_emotion": "fear"},
    {"text": "I'm not sure if it's worth continuing to shop with you.", "expected_emotion": "confusion"},
    {"text": "The service is top-notch, I am more than satisfied.", "expected_emotion": "gratitude"},
    {"text": "It frustrates me that the delivery takes so long.", "expected_emotion": "annoyance"},
    {"text": "I am saddened because my order got lost.", "expected_emotion": "sadness"},
    {"text": "Thank you very much for your help, it was very useful.", "expected_emotion": "gratitude"},
    {"text": "Your actions bring me immense joy.", "expected_emotion": "joy"},
    {"text": "The quality of service leaves much to be desired.", "expected_emotion": "disapproval"},
    {"text": "I am extremely disappointed with your service.", "expected_emotion": "disappointment"},
    {"text": "You deserve the highest praise.", "expected_emotion": "admiration"},
    {"text": "Your exchange policy terrifies me.", "expected_emotion": "fear"},
    {"text": "I am confused by your return conditions.", "expected_emotion": "confusion"},
    {"text": "Your service is simply superb, thank you.", "expected_emotion": "gratitude"},
    {"text": "The delay in delivery drives me crazy.", "expected_emotion": "annoyance"},
    {"text": "I am deeply upset about the loss of my order.", "expected_emotion": "sadness"},
    {"text": "Thanks for the support, it really helped.", "expected_emotion": "gratitude"}
]


def evaluate_model(test_messages):
    correct_predictions = 0
    correct_top5_predictions = 0
    total_predictions = len(test_messages)

    table = Table(title="Emotion Prediction Evaluation")

    table.add_column("Text", justify="left", style="cyan", no_wrap=True)
    table.add_column("Expected Emotion", justify="center", style="magenta")
    table.add_column("Predicted Emotion", justify="center", style="green")
    table.add_column("Result", justify="center")
    table.add_column("Top 5 Contains Correct", justify="center", style="yellow")

    for message in test_messages:
        response = requests.post('http://localhost:5000/predict', json={'text': message['text']})
        result = response.json()
        top5_emotions = [item['emotion'] for item in result['top_emotions']]
        predicted_emotion = top5_emotions[0]
        expected_emotion = message['expected_emotion']
        is_correct = predicted_emotion == expected_emotion
        is_correct_top5 = expected_emotion in top5_emotions

        result_color = "green" if is_correct else "red"
        table.add_row(
            message["text"],
            expected_emotion,
            predicted_emotion,
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
evaluate_model(test_messages_ru)

console.print("Evaluating model on English test messages:", style="bold blue")
evaluate_model(test_messages_en)
