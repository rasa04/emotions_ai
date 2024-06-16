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
```

Запуск сервера
```bash
uvicorn app:app --host 0.0.0.0 --port 5000
```


Тест на качества ответов
```bash
(ve) $ python ./test_answer_quality.py
```


Тест на производительность
```bash
(ve) $ python ./test_performance.py
```
