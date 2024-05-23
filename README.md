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

## Competition submission reprodction instructions
Competition submission reprodction instructions are [here: submission_reproduciton_instrucitons.md](submission_reproduciton_instrucitons.md)

## Обучение
Обучение производилось в блокноте src/train.ipynb

<!-- Перед побучением необходимо очистить тренировочный датасет -->

## Future work

## For future me
See refactoring plan [here](./Refactoring_plan.md)