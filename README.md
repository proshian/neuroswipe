# Yandex Cup 2023 ML. NeuroSwipe task

Распознавание слов по&nbsp;нарисованным кривым на&nbsp;экране смартфона (Яндекс Клавиатура)

## Task annotation
<p>В&nbsp;современном мире набор текста на&nbsp;телефоне стал неотъемлемой частью нашей повседневной рутины, и&nbsp;чтобы сделать этот процесс быстрее, разработчики мобильных клавиатур применяют различные методы. Среди них можно выделить подсказки возможных слов, автокорректор для исправления опечаток, а&nbsp;также непрерывный набор слов с&nbsp;помощью свайпа&nbsp;— тема данной задачи. Свайп&nbsp;— один из&nbsp;самых эффективных способов набирать текст на&nbsp;сенсорном экране: опытные пользователи на&nbsp;60% быстрее создают сообщения жестами, чем при вводе по&nbsp;буквам.
</p>
<p>В&nbsp;рамках этой задачи вашей целью является разработка модели, которая предсказывает слово, набранное с&nbsp;помощью свайпа по&nbsp;клавиатуре. Вам будет предоставлен доступ к&nbsp;данным о&nbsp;траектории движения пальца по&nbsp;клавиатуре (таким как координаты и&nbsp;время) и&nbsp;слове, которое хотели набрать, а&nbsp;также информация о&nbsp;раскладке клавиатуры.
</p>


## Размышления:
Есть предположение, что опытный пользователь вводит слова очень быстро и может быть не глядя. Возможно, траектории опытного и  начинающего пользователя свайп клавиатры будут отличаться

Transformer encoder for curve encoding. The input for Transformer Encoder is a sequence of vectors containing:
* x_coord
* y_coord
* dx/dt
* dy/dt
* one_hot_letter_of_nearest_keyboard_key

Производные берутся так:
1. cur_x_derivative = (next_x - cur_x) - (cur_x - prev_x); cur_y_derivative = (next_y - cur_y) - (cur_y - prev_y)
2. для первого элемента последовательности производная равна 0 или 2 * (next_x - cur_x)
3. для последнего производные совпадают с предпоследним или 2 * (cur_y - prev_y)


## Идеи из других источников:

Исследователи из google описали метод генерации синтетических данных для данной задачи в статье (Modeling Gesture-Typing Movements)[https://www.tandfonline.com/doi/permissions/10.1080/07370024.2016.1215922]
1. Ставим опорные точки в центрах клавиш, соответствуующих буквам слова
2. Добавляем шум из нормального распределения
3. Строим тракеторию, минимизирующую рывок (третью производную координаты по времени)


В статье (Phrase-Gesture Typing on Smartphones)[https://cs.dartmouth.edu/~zheer/files/PhraseSwipe.pdf] предлагается исполоьзовать предобученный Bert в качестве Encoder'а. Координаты траектории свайпа транслируются в послежовательность ближайших клавиш (то есть последовательность букв). Последовательность букв токенизируется специальными токенами Bert'а

В статье [Joint Transformer/RNN Architecture for Gesture Typing in Indic Languages](https://aclanthology.org/2020.coling-main.87.pdf) в качестве энкодера multihead self attention -> BiLSTM. На входе последовательность векторов [x_coord, y_coord, dx/dt, dy/dt, one_hot_letter_of_nearest_keyboard_key]

## Вопросы
* Как токенизировать выходную последовательность? Пока планирую побуквенно.
* Если входную последовательность использовать как [x_coord, y_coord, dx/dt, dy/dt, one_hot_letter_of_nearest_keyboard_key], не нужно ли заменить one_hot на embedding?

## TODO
Urgent
* Энкодер, принимающий на вход координаты
Super important
* Synthetic data generation
    * WordGesture-GAN
    * Modeling Gesture-Typing Movements (minimal jerk)


## Synthetic data generation:
### Minimal-jerk
1. Берем центры букв
2. Сдвигаем с помощью двумерного гаусовского распределения
3. Берем готовую реализацию minimal-jerk trajectory

Вопросы:
1. Есть ли у minimal-jerk параметры?
2. Должны ли minimal-jerk trajectory алгоритм знать сколько времени прошло между via-points?
2. Как от кривой-результата алгоритма получить последовательность точек с временным интервалом t?

Ошибку попадания в клавишу можно моделировать гаусовким распределением ([Predicting Finger-Touch Accuracy Based on the Dual Gaussian Distribution Model](https://dl.acm.org/doi/pdf/10.1145/2984511.2984546))

$$
\begin{align}
σ^2_x = 0.0075W^2_x + 1.68 \\
σ^2_y = 0.0108W^2_y + 1.33\\
µ_x = µ_y = 0
\end{align}
$$

Где $W_x$ - размер клавиши по $x$, $W_y$ - размер клавиши по $y$.

## Subtasks
* baseline creation
* finding papers
* reading papers
* finding datasets
* creating dataset
