# NeuroSwipe

A transformer neural network for a gesture keyboard that transduces curves swiped across a keyboard into word candidates.

The repository is my yandex cup 2023 solution (7-th place) with improvements

## Demo

You can try out one of the models trained as part of the competition in a [web app](https://proshian.pythonanywhere.com/)


![demo](https://github.com/proshian/neuroswipe/assets/98213116/79b3506d-2817-45b5-9459-92490edcd5dc)


If the website is not available, you can run the demo yourself by following the instructions from [the web app's GitHub repository](https://github.com/proshian/neuroswipe_inference_web).


## Existing work
1)  Alsharif O. et al. Long short term memory neural network for keyboard gesture decoding //2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). – IEEE, 2015. – P. 2076-2080.
2) Xu Z. et al. Phrase-Gesture Typing on Smartphones //Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology. – 2022. – P. 1-11.
3) Biju E. et al. Joint transformer/RNN architecture for gesture typing in indic languages // Proceedings of the 28th International Conference on Computational Linguistics. – 2022 – P. 999–1010.
4) Jia N., Zhang Y., Wu Q. Enhancing Swipe Typing with Gated Linear Transformer //2024 International Conference on Emerging Smart Computing and Informatics (ESCI). – IEEE, 2024. – P. 1-6.


<br>



<table width="1068">
<tbody>
<tr>
<td width="174">
<p>Method</p>
</td>
<td width="89">
<p>Year</p>
</td>
<td width="177">
<p>Model</p>
</td>
<td width="147">
<p>Model type</p>
</td>
<td width="367">
<p>Swipe point embedding</p>
</td>
<td width="113">
<p>Params num</p>
</td>
</tr>
<tr>
<td width="174">
<p>LSTM for KGD (1)</p>
</td>
<td width="89">
<p>2015</p>
</td>
<td width="177">
<p>Peephole BiLSTM</p>
</td>
<td width="147">
<p>Encoder-only</p>
</td>
<td width="367">
<pre><code>concat(
   nearest_key_one_hot,
   x, y, t)
</code></pre>
</td>
<td width="113">
<p>1.5</p>
</td>
</tr>

<tr>
<td width="174">
<p>Phrase Gesture Typing (2)</p>
</td>
<td width="89">
<p>2022</p>
</td>
<td width="177">
<p>Encoder: Bert Base,</p>
<p>Decoder: Bert Base</p>
</td>
<td width="147">
<p>Encoder-decoder</p>
</td>
<td width="367">
<pre><code>nearest_key_embedding
</code></pre>
</td>
<td width="113">
<p>220</p>
</td>
</tr>

<tr>
<td width="174">
<p>Joint Transformer/</p>
<p>RNN for Indic Keyboard (3)</p>
</td>
<td width="89">
<p>2022</p>
</td>
<td width="177">
<p>Encoder: Transformer + GA + LSTM;</p>
<p>Contrastive distance: ELMo</p>
</td>
<td width="147">
<p>Encoder-only и Contrastive Distance</p>
</td>
<td width="367">
<pre>concat(
   nearest_key_one_hot,
   x, y, dx/dt, dy/dt)
</pre>
</td>
<td width="113">
<p><b>TO BE COUNTED</b></p>
</td>
</tr>
</tbody>
</table>

It can be seen on the table above that all existing approaches use similar swipe point embeddings based on the embedding of the nearest key, and other options have not been explored.

At the same time, a hypothesis arises that including information about all keys on the keyboard in the embeddings can mitigate the noise inherent to this task and more accurately reflect the user's interaction with the keyboard. Thus the research presented in this repository mainly focuses on swipe point representations and their effect on metrics.


## Method

### Model

The model is an encoder-decoder transformer with hyperparameters:

|        Hyperparameter        | Value |
| ---------------------------- | ----- |
| Encoder layers number        | 4     |
| Decoder layers number        | 4     |
| Model Dimension              | 128   |
| Feedforward layer dimension  | 128   |
| Encoder heads number         | 4     |
| Decoder heads number         | 4     |
| Activation function          | ReLU  |
| Dropout                      | 0.1   |

All experiments utilize this exact model and the primary difference between the experiments is in the swipe-dot-embedding-layer (embeddings that are input to the encoder).

Encoder input is a sequence of `swipe point embeddings`. They are described in a dedicated section.

Decoder input is a sequence of trainable embeddings (with positional encoding) of tokens extracted from the target word. In this case okens are all alphabet charracters and special tokens (`<sos>, <eos>, <unk>, <pad>`) 

The positional encodeing is the same as in "Attention is all you need": it's a fixed embedding based on harmonic oscilations.

### Other models

* There were experiments where first tranformer encoder layer can input a sequence with elements of a dimension different from other encoder layers. Actually, I used this kind of custom transfomer in yandex cup, but I don't the difference in performance or parameters economy substential.
* There was an expermient with a larger model: it seems like there is a great potential, but it's too expensive to train for me.


### Swipe point embeddings

#### My nearest SP embedding

Encoder input sequence consists of elements denoted as `swipe point embedding` on the image below.

![Here should be an image of encoder_input_sequence_element](./REAME_materials/encoder_input_sequence_element.png)

The $\frac{dx}{dt}$, $\frac{dy}{dt}$, $\frac{d^2x}{dt^2}$, $\frac{d^2y}{dt^2}$ derivatives are calculated using finite difference method.

Keyboard key embeddings used in encoder and charracter embeddings used in decoder are different entities.

#### My weighted SP embedding

![weights_viz](https://github.com/user-attachments/assets/d2c2505e-91c8-4c33-8bcc-85c386441628)
![3d_keyboard](https://github.com/user-attachments/assets/dce6b76b-b635-4f93-9251-48dac5e4d793)

## Some info can be found in solution_description.md

More info in [solution_description.md](solution_description.md) file (in Russian and may be outdated).


## Results


> [!NOTE]  
> The sudden metrics improvement for `my_nearest_features` and `indiswipe_features` is due to a decrease in learning rate. The  losses on validation set of these models did not change over 2000 validations, which led to the ReduceLROnPlateau scheduler cutting learning rate in half. For other models, ReduceLROnPlateau haven't taken any actions yet.


![beamsearch_metrics](https://github.com/user-attachments/assets/cddb1290-8886-4eb4-9366-f072e191e3fc)

Greedy decoding word level accuracy (train set) | CE loss (train set)
:-------------------------:|:-------------------------:
![acc_greedy_TRAIN](https://github.com/user-attachments/assets/b7eda630-b007-442b-b34d-825bb0cd80c4)  |  ![celoss_TRAIN](https://github.com/user-attachments/assets/d801022d-a454-4916-8102-84ec0e228446)

Greedy decoding word level accuracy (validation set) | CE loss (validation set)
:-------------------------:|:-------------------------:
![acc_greedy_val](https://github.com/user-attachments/assets/30dee9b6-a55e-4760-a9fb-9c3ecab4fa79) | ![celoss_val](https://github.com/user-attachments/assets/0f161cdd-8dd9-44bc-b93f-ca6504ea7956)

token level accuracy (validation set) | token level f1-score (validation set)
:-------------------------:|:-------------------------:
![acc_token_val](https://github.com/user-attachments/assets/4ff2a463-1ac8-475c-8ec9-187e821ed229) | ![f1_token_val](https://github.com/user-attachments/assets/5925c80f-4d46-4758-9a56-828f2f7cfe3e)



## Prerequisites

Install the dependencies:

```sh
pip install -r requirements.txt
```

* The inference was tested with python 3.10
* The training was done in kaggle on Tesla P100


## Training

<!-- Перед побучением необходимо очистить тренировочный датасет -->

The training is done in [train.ipynb](src/train.ipynb)

> [!WARNING]  
> `train.ipynb` drains RAM when using `n_workers` > 0 in Dataloader. This can result in up to `dataset_size * n_workers` extra gigabytes of RAM usage. This is a known issue (see [here](https://github.com/pytorch/pytorch/issues/13246)) that happens when dataset uses list to store data. Although `torch.cuda.empty_cache()` can be used as a workaround, it doesn't seem to work with pytorch lightning. It appears I didn't commit this workaround, but you can adapt train.ipynb from [before-lightning branch](https://github.com/proshian/neuroswipe/tree/before-lightning) by adding ```torch.cuda.empty_cache()``` after each epoch to to avoid the issue.


## Prediction

[word_generation_demo.ipynb](src/word_generation_demo.ipynb) serves as an example on how to predict via a trained model.

[predict_v2.py](src/predict_v2.py) is used to obtain word candidates for a whole dataset and pickle them

> [!WARNING]  
> If the decoding algorithm in `predict_v2.py` script utilizes a vocabulary for masking (`use_vocab_for_generation: true` in the config), it is necessary to disable multiprocessing by passing the command-line argument `--num-workers 0` to the script. Otherwise, the prediction will take a long time.




## Yandex cup 2023 submission reproduction instructions
Yandex cup 2023 submission reprodction instructions are [here: submission_reproduciton_instrucitons.md](submission_reproduciton_instrucitons.md)




## For future me
See [refactoring plan](./Refactoring_plan.md)