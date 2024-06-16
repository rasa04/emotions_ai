from datasets import load_dataset
from transformers import BertTokenizer

# Загрузка датасета с указанием конфигурации 'raw'
dataset = load_dataset('seara/ru_go_emotions', 'raw')

# Проверка доступных разделов
print(dataset)


# Функция для создания меток
def create_labels(example):
    emotion_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
                      'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
                      'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
                      'remorse', 'sadness', 'surprise', 'neutral']
    labels = [example[emotion] for emotion in emotion_labels]
    example['labels'] = labels.index(1) if 1 in labels else 27  # Set to neutral (27) if no emotion is labeled
    return example


# Применение функции создания меток к датасету
dataset = dataset.map(create_labels)


# Объединение текстов на русском и английском языках
def combine_texts(example):
    example['combined_text'] = example['ru_text'] + " " + example['text']
    return example


dataset = dataset.map(combine_texts)

# Разделение на обучающую и тестовую выборки (80/20)
train_test_split = dataset['train'].train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']


# Преобразование данных в формат, подходящий для обучения
def preprocess_data(examples):
    return tokenizer(examples['combined_text'], truncation=True, padding='max_length', max_length=128)


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
train_dataset = train_dataset.map(preprocess_data, batched=True)
test_dataset = test_dataset.map(preprocess_data, batched=True)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels'])

train_dataset.save_to_disk('train_dataset')
test_dataset.save_to_disk('test_dataset')

print("Данные успешно подготовлены и сохранены")
