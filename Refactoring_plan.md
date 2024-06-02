* Мб перейти с моего паттерна THING_NAME_TO_CTR[config['thing_name']](config['thing_kwargs']) на hydra

# Перед мерджем в мейн:

* Описать эксперименты и результаты:
      * nearest (my)
      * weighted
      * nearest (indic)
      * nearest (lstm 2015)
* Добавить пометку про обучение с pl (ест память)
* Добавить альтернативный блокнот с обучением (который не ест память)
* Добавть пометку, что нельзя использовать многопоточку, если генерация слов производится с использванием словаря 



# После окончания
* Переписать все модели в виде v2_model, описать абстрактуную версию для пояснения структуры всех моеделей, также пояснить, что kwargs зло, поэтому абстрактно сделать не мог
* А лучше сделать Encoder и Decoder модели с унифицированным интерфейсом!!! Это будет TransformerLikeEncoderDecoder. Получает на вход классы, как текущаая имплементация AbstractEncoderDecoder, но вызывает encoder и decoder с конкретными параметрами. Нужно просто добиться, чтобы любой класс из Transformer-Like имел одинаковый интерфейc. То есть TransformerEncoder и ConformerEncoder имели одинаковые интерфейсы; Также будет RNNLikeEncoderDecoder
* Далее все эти модели нужно обучить по-новой или убедиться, что состояние pl подгружается





## Воспроизводимость yandex-cup
Очень странно, но посылка yandex-cup не воспроизводится из-за типа данных токенов букв слова. Раньше был int32, сейчас int64. Если вернуть int32, воспроизведется. Кажется, что тип данных не должен играть никакую роль, потому что все токкены вписываются даже в int1.

Перехдл к int64 был сделан, потому что что CELoss не поддерживает int32 и при обучении нужно кастить к int64, если тип не тот

Нужно проверить, как лучше предсказания с int32 или int64

Может быть, нужно зарепортить баг в pytorch

Проверить, вдруг тип данных эмбеддингов зависит от типа данных токенов. Вдруг, например, если токены int32, то эмбеддинги float32, а если int64, то эмбеддинги float64

-----------------



* Replace SwipeCurveTransformerDecoderv1 with Transformer Decoder




Проблема в predict.py:
! Если use_vocab_for_generation == True 
многопоточность почему-то сильно медленнее, чем выполнение в главном потоке.
Поэтому num_workers должно быть равно 0 (это запуск без многопоточности).
Предполагаю, что замедление связано с переносом из одного потока в другой
очень большого словаря, хранящегося в генераторе слов



# Refactoring Plan

Мб добавить в beamsearch:
* Пропускание токена, если его вероятность `<=` какого-то значения (по дефолту `float(-int)`)
* Мб добавить удаление потенциального кандидата слова (последовательности букв), если его вероятность ниже какого-то значения (по дефолту `float(-int)`)

* Возможно стоит переименовать `create_ds_on_disk.py` в `save_datalist_kwargs.py` и сохранять на диск не только data_list, но и `get_item_transform` и `grid_name_list`. Создание датасета из данной репрезентации будет таким `CurveDataset.from_datalist(**torch.load(path))`

* Добавить пост-обработку предсказаний с помощью расстояния Левиштейна. Тут важно для всех слов с минимальным расстоянием Левинштейна к вероятным, но ненастоящим словам померить моделью вероятность и переранжировать.

* Оценить, насколько beamsearch дешевле обхода всего словаря
* Сделать совмещение beamsearch и обхода словаря: сначала несколько шагов beamsearch. Дальше оцениваем вероятности всех слов с префиксами, полученными c beamsearch

* Написать новый trainer
* Изучить torch_lightning trainer


* Сделать jupyter notebook с оценкой времени создания и времени итерирования для CurveDataset с разными вариантами transform
* Создать extra_train_ds, который сразу хранит все в должном виде
* Попробовать создать default_train_ds, который сразу хранит все в должном виде
      * Если не получится, пусть все токены хранятся как uint8 или int16 (подсмотреть в array версии, вспомнить почему такие ограничения), а в get_item_trainsform переделываются в int32
* Проверить время одной эпохи после использования этих новых датасетов


* перенести тестирвоание collate_fn из train.ipynb в unittest/test_collate_fn.py
* обновить существующие тесты
* Сделать CI github action, запускающий юнит-тесты

1) Добавить сохранение таблицы в predict.py -> save_results
      * В predict.py определить словарь word_generator_name__to__kwarg_keys: Dict[str, Set[str]], отображающий название алгоритма декодирования в множество, которому должно быть равно word_generator_kwargs.keys(). На основе этого словаря необходимо определить пред-условие для выполнения скрипта: `assert word_generator_name__to__kwarg_keys[word_generator_name] == set(config['word_generator_kwargs'])`
      * это необходимо, потому что word_generator_kwargs, переданный скрипту может быть неисчерпывающим (если у алгоритма есть параметры по умолчанию) или наоборот избыточным (например, 'verbose': false). Обе стуации могут привести к проблемам: в случае избыточности два одинаковых predictor'а будут сочтены за разные, в случае недостаточности, наоборот, два разных predictor'а могут быть сочтены за один. Речь идет об этапе поиска predictor'а в таблице по уникальному сочетанию 'model_name', 'model_weights', 'generator_name', 'grid_name', *[f'gen_name__{k}' for k in generator_kwargs.keys()] c wелью дальнейшего заполнения пути до произведенного предсказания
      * Есть ПЛОХАЯ альтернатива, которая решает проблему неисчерпываемости, но не решает проблему избыточности (а скорее усугубляет) - дополнять kwargs значениями по умолчанию с помощью inspect.signature(func)
      
2) Убедиться, что предсказание + аггрегация выдают тот же результат, что и submission



* может быть, нужен pandas MultiIndex

* Наладить связь между predict.py и aggregate_predictions.py
      * predict.py должен возвращать таблицу с predictior_id, Generator_type, Generator_call_kwargs_json, Model_architecture_name, Model_weights_path, Grid_name, test_preds_path, val_preds_path, validation_metric

      * В таблице хранить generator_kwargs как отдельные столбцы!! формируется так: f"{generator_name}_{kwarg_name}" -> kwarg_val

      * aggregation.py должен иметь аггрегаторы, каждый из которых имеет fit и predict, а также хранит всю информацию о predictor'ах. Fit производится на valid части, predict - на test части. Init агрегатора получает на вход таблицу и нужные redictor_id (по умолчанию все id) 

      * Возможно, Нужен utility, объединяющий несколько таблиц, если предсказания делались на разных машинах. Predicotr_id должен удаляться, далее обходим все уникальные сочетания (Generator_type, Generator_call_kwargs_json, Model_architecture_name, Model_weights_path, Grid_name). сочетание представлено более чем одной строчкиой, убеждаемся, что для каждого из столбцов {test_preds_path, val_preds_path, validation_metric} мощность объединения значений по этим строчкам не превышает 1. Если это так производим объединение и записываем в одну из строчек, остальные удаляем. Иначе вызываем ошибку. В конце переназначаем id и перезаписываем файл.


* Добавить скрипт для получения метрики для предсказания
* Убедиться, что greedy_search + этот скрипт работают
* Заменить все, что связано с оценкой моделей в playground на вызов predict.py + evaluate.py (скорее на вызов функций из этих скриптов)


* Можно вообще не иметь id. Организовать базу данных либо в виде pandas таблички, либо такого словаря: tuple(model name, model path, grid_name, generator-name, tuple(sorted(kwargs.items()))) -> all_results_dict = {path to validation results, path to test results}


* Сделать batch_first ветку, обучить там транфсормер, у которого dim encoder целиком равен dim decoder


* Обновить word_generation.ipynb. 
      1. Генерация с помощью greedy или beam generator
      2. Предсказание для датасета с помощью Predictor(generatr="greedy")
      3. Рассчет mmr для результата predictor
      4. Рассчет mmr для предсказаний, сохраненных в файлах
      5. Аггрегация
      6. Рассчет mmr для аггрегированных предсказаний


* перенести get_grid_name_to_grid из predict в dataset_utils

# Notes about dataset

Input_transform controls what would be stored in `dataset.data_list`. Tuples of `x, y, t, gridname, target` are stored if no input_transform is given.  

`get_item_transform(input_trasform(x, y, t, gridname, target)) = model_input, target`. 

Currently `dataset.data_list` stores `(x, y, t, gridname, target, kb_tokens)`.

I tried:
* storing `(x, y, t, gridname, target)` and calculating all featurs on `__getitem__`. Getting items became to slow. Iterating over the dataset takes 4 times more time compare to storing `(x, y, t, gridname, target, kb_tokens)`
* storing `model_input, target`. Creating dataset became too slow (around 5 hours)



Выделение nearest_key_loookup в отдельную сущность хорошая идея, потому что:  
* Во-первых, зечем тратить каждый раз 5-10 секунд на создание маппинга [координаты → лейбл ближайшей буквы], если можно одина раз создать, сохранить и всегда передавать.
* Эта сущность нужна на инференсе

Большим плюсом датасета, хранящего `(x, y, t, gridname, target)` является возможность добавления аугментаций датасета (случайное смещение координат x, y)



# General

Обученная модель = 
* Имя раскладки
* Путь до весов
* Имя архитектуры
* Имя способа предобработки данных

Нужно ли делить json на отдельные grid'ы?

Если я уж разделил все датасеты на отдельные grid'ы можно валидационной и тестовой выборке добавить отдельный файл original indexes, чтобы удобно производить валидацию и удобно делать submission.


# Prediction

При предсказании предлагается заполнять таблицу prediction_table.csv:
* word_generator_id aka predictor_id
      * (определяется всем, кроме test_preds_path, val_preds_path, validation_metric)
* –
* Generator_type
* Generator_call_kwargs_json
* Model_architecture_name
* Model_weights_path
* –
* Grid_name
* –
* test_preds_path
* val_preds_path
* validation_metric


# Aggregation
Скрипт агрегации получает на вход
* состояние агрегатора
* prediction_table.csv


Аггрегатор имеет
* Метод load_state(state)
      * state всегда ссылается на predictor_id. В случае взвешенной аггрегации state – это словарь predictor_id__to__weight
* Метод get_predictor_ids()
* Метод aggregate(predictions_dict)
      * Проверяет, что все predictor_id из его состояния есть в predictions_dict.keys()
      * Агрегирует :)
      * В случае взвешенного агрегатора:
            * scaled_preds = []
            * for predictor_id, weight in predictor_id__to__weight.items():
                  * scaled_preds.append(scale_preds(predictions_dict['predictor_id'], weight))
            * final_preds = merge_sorted_preds(scaled_preds)


predictions_dict будет поучаться с помощью функции get_predictions_dict(predictor_ids, dataset_split), которая по списку id и типу датасета (train/val) находит их в табличке и возвращает словарь {id → предсказания}



# Представление предсказаний
Сейчас я вижу три варианта представдения предсказаний: 
1. список предсказаний длиной с исходный датасет. Для строк с расскаладками,
      с которыми модель не умеет работать она предсказывает пустой список.
2. список предсказаний длиной с число элементов данной раскладки.
3. словарь длиной с число элементов данной раскладки 
      `индекс в исходном датасете` -> список предсказаний для данной кривой.
      Сюда же отнесу вариант, где предсказания представлены списком, каждый
      элемент которого - кортеж (номер_строки_в_исходном_датасете, список_предсказаний).
Также может быть добавлена метаинформация:
* название архтиектуры
* путь к весам
* название раскладки
* список индексов

Вариант, когда каждая модель выдает все 10_000 строк кажется
не совсем удобным: Когда мы подбираем наилучшие гиперпараметры
для аггрегации, удобно оценивать гиперпараметры по результатам
метрик именно на данной раскладке.


# Other
* Авторегрессионное использование трансформера сопровождается огромным числом повторных вычислений. Когда мы предсказываали предыдущий токен мы уже посчитали все quiries, keys и values на всех слоях для всех токенов, кроме последнего. Добавление кэширования может вдвое ускорить скорость предсказания 


# Notes 
* Все, что многопоточное (Predictor._predict_raw_mp и CurveDataset._get_data_mp) работает только в скриптах. В jupyter notebook'ах - нет. Кажется, проблема в использовани concurrent.futures. Возможно, все наладится, если исопльзовать модуль multiprocessing.