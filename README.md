# Yandex Cup 2023 ML. NeuroSwipe task

Распознавание слов по&nbsp;нарисованным кривым на&nbsp;экране смартфона (Яндекс Клавиатура)

## Как воспроизвести последнюю посылку:

1. Основные наборы данных распаковать в `./data`. То есть, например, путь до train.json: `.data/data/result_noctx_10k/train.jsonl`

2. Получить предсказания оффициального бейзлайна:

```shell
python ./src/keyboard_start/lib/main.py --train-path data/data/result_noctx_10k/train.jsonl --test-path data/data/result_noctx_10k/test.jsonl --voc-path data/data/result_noctx_10k/voc.txt --num-workers 4 --output-path ./result.csv
```

3. Получить датасет в другом формате: в каждой строке каждого .jsonl файла 'grid' заменен на 'grid_name', а соответствие 'grid_name_to_grid.json' сохранено в отдельный файл. Такой датасет должен быть сохранен в директории ./data/data_separated_grid. Для этого можно запустить скрипт ниже. В качестве альтернативы можно скачать содержимое ./data/data_separated_grid c [гугл диска](https://drive.google.com/drive/folders/1rRBUKUC0D6eZBJqT9qKs5fKQLl-gboej?usp=sharing).

```shell
python ./src/separate_grid.py
```

4. Загрузить чекпойнты весов моделей, используемых в последней посылке из [гугл диска](https://drive.google.com/drive/folders/1-iFPYCcRYy-tEu14Ry6xU6SMMf3eCjn6?usp=sharing) в папку [./data/trained_models_for_final_submit/](./data/trained_models_for_final_submit/).

5. Получить предсказания для каждой отдельной модели. Для этого запускаем из корня директория скрипт:

```shell
python ./src/get_individual_models_predictions.py
```

6. Агрегировать предсказания с помощью скрипта ...

## Обучение
Обучение производилось в блокноте .... 

Перед побучением необходимо очистить тренировочный датасет