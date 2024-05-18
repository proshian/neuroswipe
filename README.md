# Yandex Cup 2023 ML. NeuroSwipe task

Распознавание слов по&nbsp;нарисованным кривым на&nbsp;экране смартфона (Яндекс Клавиатура)

## Demo

You can try out one of the models trained as part of the competition in a [web app](https://proshiang.pythonanywhere.com/)


![demo](https://github.com/proshian/yandex-cup-2023-ml-neuroswipe/assets/98213116/1c7210ab-f347-4f50-9105-c6eb74884827)


## Method

The model is encoder-decoder transformer.
The first tranformer encoder layer can input a sequence with elements of a dimension different from other encoder layers.

Encoder input sequence consists of elements denoted as `swipe point embedding` on the image below.

![Here should be an image of encoder_input_sequence_element](./REAME_materials/encoder_input_sequence_element.png)

The $\frac{dx}{dt}$, $\frac{dy}{dt}$, $\frac{d^2x}{dt^2}$, $\frac{d^2y}{dt^2}$ derivatives are calculated using finite difference method.

Decoder input sequence consists of character-level embeddings (with positional encoding) of the target word.

Keyboard key embeddings used in encoder and charracter embeddings used in decoder are different entities.

More info in [solution_description.md](solution_description.md) file (in Russian).


## Как воспроизвести последнюю посылку:

1. Установить зависимости:

```shell
python -m pip install -r sumbission_reproduction_requirements.txt
```

2. Скачать и распаковать Основные наборы данных в `./data`. То есть, например, путь до train.json: `.data/data/train.jsonl`. Скачать и распаковать основной датасет можно скрипта ниже (арихив сохранен не будет):

```shell
python ./src/downloaders/download_original_data.py
```

3. Получить предсказания оффициального бейзлайна:

```shell
python ./src/keyboard_start/ks_lib/main.py --train-path data/data/train.jsonl --test-path data/data/test.jsonl --voc-path data/data/voc.txt --num-workers 4 --output-path ./results/submissions/baseline.csv
```

4. Получить датасет в другом формате: в каждой строке каждого .jsonl файла 'grid' заменен на 'grid_name', а соответствие 'grid_name_to_grid.json' сохранено в отдельный файл. Такой датасет должен быть сохранен в директории ./data/data_separated_grid. Для этого можно запустить скрипт ниже. 

```shell
python ./src/separate_grid.py
cp ./data/data/voc.txt ./data/data_separated_grid/voc.txt
```

В качестве альтернативы можно скачать результаты работы скрипта `./src/separate_grid.py` c [гугл диска](https://drive.google.com/drive/folders/1rRBUKUC0D6eZBJqT9qKs5fKQLl-gboej?usp=sharing). в `./data/data_separated_grid `. Это можно сделать, запустив скрипт ниже:

```shell
python ./src/downloaders/download_dataset_separated_grid.py
```

5. Загрузить веса моделей, используемых в последней посылке из [гугл диска](https://drive.google.com/drive/folders/1-iFPYCcRYy-tEu14Ry6xU6SMMf3eCjn6?usp=sharing) в папку [./results/final_submission_models/](./data/trained_models_for_final_submit/). Это можно сделать, запустив скрипт ниже:

```shell
python ./src/downloaders/download_weights.py
```

6. Получить предсказания для каждой отдельной модели. Для этого запускаем из корня директория скрипт:

```shell
python ./src/predict.py --num-workers 4 --config ./config-yandex-cup.json
```

В результате директория ./results/final_submission_predictions/test наполнится pickle файлами с предсказаниями

7. Агрегировать предсказания:

```shell
python ./src/aggregate_predictions.py
```

В результате в директории ./results/submissions будет создан файл my_last_submission.csv, который был моим финальным посылом.


## Обучение
Обучение производилось в блокноте src/train.ipynb

<!-- Перед побучением необходимо очистить тренировочный датасет -->

## Future work

## For future me
See refactoring plan [here](./Refactoring_plan.md)