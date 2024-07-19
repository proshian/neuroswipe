# Вопросы к авторам
1) Почему в генерации датасета (например, Hindi) один ряд состоит из 11 букв, одна клавиша имеет ширину 3, но клавиатура создается 30x15, а не 33x15?
2) Как делили датасет на train - val - test? Например, в статье написано, что 70% train, а в коде - 60%. Где правда? В идеале, конечно, было бы здорово test сет выделить прямо в xlsx, чтобы было понятно как сравниваться

# Note

Так как по всей видимости, код уних в репозитории некорректный есть 2 пути: 
1) указать на ошибки и спросить как правильно. Тогда можно будет любые фичи использовать
2) Использовать их датсет без преобразования, но тогда можно использовать только их фичи.
    * А все равно не понятно как сравниваться, потому что не понятно, как делили на train, val и test)))

# Dataset structure

IndicSwipe датасеты содержат (tgt_word), x, y, x', y', keyboard_key_id


# Task 1
Convert the indic datasets so that 
* has an element of a dataset is a tuple: (x, y, t, `dataset_name`, tgt_word)
* has a grid (keyboard layout) in `grid_name_to_grid.json` similar to ones in yandex dataset.
* has a voc.txt with all possible words for the lanuage (if not provided explicitly it is just a set of all words in the dataset; in a general case words in the dataset are a subset of voc.txt)


`grid_name_to_grid.json` contains a dict grid_name -> grid where a grid defines a keyboard layout. 
Each grid has width and height properties (keyoard width and keyboard height) and a keys property that is a list of keys. 
Each key has 
* lablel or action property with label (letter the a key) or action (what this key does (space, shift etc.))
* x, y - upper left corner
* w, h - width and height off the key


# Questions
I started with Hindi dataset. It seems that all datasets are very similar

The grid is defined with ...

## `gesture_path_generation_hindi.py`
* why there are 11 chars per row, each key's width = 3, but `keyboard_full_size` has width 30?
* Also the embeddings are clipped so that each cordinate is between 0 and 29, but we can see wxamples with x coord over 30 in the dataset?

**Seems like the code is wrong**



# Состояние:

Time is constatnly 0.5 miliseconds
I made this conclusion because:
```
x_derivative = traj_x[i+1]-traj_x[i-1]
y_derivative = traj_y[i+1]-traj_y[i-1]
```

We can just sanity check the dataset that for each element 

```python
t = [0.5] * list(range(1, seq_len + 1))
for i in range(1,seq_len):
    t[i+1] - t[i-1] == (x[i+1] - x[i-1]) / dx_dt[i] == (y[i+1] - y[i-1]) / dy_dt[i]
```



