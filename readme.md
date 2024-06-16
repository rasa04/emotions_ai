***Нейронка для определения эмоций по сообщению***


Прежде чем работать с моделью нужно создать виртуальное окружение

```bash
python -m venv ./ve
```

Дальше активировать это окружение и установить зависимости
```bash
$ ./ve/Scripts/activate

(ve) $ pip install -r requirements.txt
```

Загрузка датасета
```bash
(ve) $ python ./load_dataset.py
```

Тренировать модель
```bash
(ve) $ python ./train_model.py
(ve) $ python ./train_model.py --checkpoint ./results/checkpoint-31500
```

Запуск сервера
```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```


Пример запроса в сервер нейронки
```bash
curl --location 'http://127.0.0.1:5000/predict' \
--header 'Content-Type: application/json' \
--data '{
    "text": "Today is an incredible day! Thank you for the excellent service."
}'
```


Тест на качества ответов
```bash
(ve) $ python ./test_answer_quality.py
```


Тест на производительность
```bash
(ve) $ python ./test_performance.py
```
