**Основное:**

Датасет принимает на вход initial_transform и get_item_transform. Подробнее написано в классе датасета. Обучение производилось с initial_transform = InitTransform, get_item_transform = GetItemTransform из файла transforms


Кажется, многопоточный dataloader поздравляет вынести выделение всех фичей в getitem с нулевыми накладными расходами на эти преобразования.
Такой подходеще прекрасен тем, что можно добавить аугментацию в виде прибавления случайного шума к координатам (например randint (-2, 2)), и модель будет меньше переобучаться.



**Разростание RAM**
https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662

Как тут описано, RAM бурно растет при использовании многопоточки в даталоадере. Точнее за каждую эпоху RAM увеличивается вдвое. Благо сразу после эпохи можно вызвать torch.cuda.empty_cache() и откатить объем обратно.

Чтобы этого не происходило нужно отказаться от хранения данных в огромном списке и заменить, например, на numpy array, но НЕЛЬЗЯ dtype=object. У меня датасет - это список кортежей (от кортежей тоже нужно отказаться) массивов разной длины. Различная длина, похоже, не позволяет положить их всех в один numpy array. Может, можно сделать массив указателей на array и положить туда, но скорее всего, память все равно будет разрастаться (это уже не будет один сплошной блок памяти с единственной ссылкой). Еще может быть поможет преобразование  DataFrame. 
Но меня пока устраивает решение с torch.cuda.empty_cache()


**Возможно, если датасет разрастется поэявится резон перейти к такому формату, чтобы не переполднять RAM**

``` python
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y
```

Только list_IDs лучше сделать numpy array'км, а не списком. Потому что иначе память будет разростаться при многопоточном dataloader'е.


**Подробные результаты:**

BATCH_SIZE = 320
max_batches_per_epoch_train=2000

Время на загрузку датасета:
* загрузить train_default_grid_no_errors__2023_10_31_ft__uint8, сохраненный в bin’ах по 10_000 элементов занимает 47 минут. Штука в том, что такой датасет влезвает в RAM, но RAM преполняется при попытки сохранить датасет. Так вот создать датасет - примерно 45 минут. Поэтому BinListStorage - вообще бессмысленная затея
* Загрузить TrajFeatsKbTokensTgtWord_{transform_type} (загрузка сохраненного datalist) - примерно 7 минут

Время на создание датасета:
* создать train_default_no_errors_uint8_datalist__TrajFeats_KbTokens_TgtWord – 36 минут
* Как я помню, на создание fully transformed default датасета с нуля требуется 45 минут а kaggle


Длительность эпохи (то есть 2000 батчей по 320 сэмплов) при num_workers = 0:
* all_transforms_in_getitem - 16 минут
* KbTokens_InitTransform+KbTokens_GetItemTransform –  10 минут 50 секунд (650 секунд)
* fully transformed – примерно 7 минут 15 секунд (438)
* TrajFeatsKbTokensTgtWord_{transform_type} - примерно 8 минут


Если использовать num_workers = 4
* all_transforms_in_getitem - 7 минут 10 секунд
* TrajFeatsKbTokensTgtWord_{transform_type} -7 минут 10 секунд

В случае с TrajFeatsKbTokensTgtWord_{transform_type} на самом деле, эпоха не была окончена из-за переполнения RAM (о том, что  многопоточка даталоадеров, если в датасете есть pure python objects может переполнять память написано в документации), но последнее, что вывел tqdm было:
91%|█████████▏| 1826/2000 [06:38<00:35,  4.88it/s]


Это не ошибка. При num_workers = 4 получается одно время у all_transforms_in_getitem и TrajFeatsKbTokensTgtWord_{transform_type}