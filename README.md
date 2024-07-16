# NeuroSwipe

A transformer neural network for a gesture keyboard that transduces curves swiped across a keyboard into word candidates.

This repository used to contain my Yandex Cup 2023 solution (7th place), but after many improvements, it has become a standalone project.

## Demo

You can try out one of the models trained during the competition in a [web app](https://proshian.pythonanywhere.com/)


![demo](https://github.com/proshian/neuroswipe/assets/98213116/79b3506d-2817-45b5-9459-92490edcd5dc)

If the website is not available, you can run the demo yourself by following the instructions in [the web app's GitHub repository](https://github.com/proshian/neuroswipe_inference_web).


## Existing Work
1)  Alsharif O. et al. Long short term memory neural network for keyboard gesture decoding //2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). – IEEE, 2015. – P. 2076-2080.
2) Xu Z. et al. Phrase-Gesture Typing on Smartphones //Proceedings of the 35th Annual ACM Symposium on User Interface Software and Technology. – 2022. – P. 1-11.
3) Biju E. et al. Joint transformer/RNN architecture for gesture typing in indic languages // Proceedings of the 28th International Conference on Computational Linguistics. – 2022 – P. 999–1010.
4) Jia N., Zhang Y., Wu Q. Enhancing Swipe Typing with Gated Linear Transformer //2024 International Conference on Emerging Smart Computing and Informatics (ESCI). – IEEE, 2024. – P. 1-6.


<br>

Methods comparison

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
<p>Encoder-only and Contrastive Distance</p>
</td>
<td width="367">
<pre>concat(
   nearest_key_one_hot,
   x, y, dx/dt, dy/dt)
</pre>
</td>
<td width="113">
<p><b>TODO: FILL THIS FIELD</b></p>
</td>
</tr>

<tr>
<td width="174">
<p>Linear Transformer (4)</p>
</td>
<td width="89">
<p>2024</p>
</td>
<td width="177">
<p>Gated Linear Transformer</p>
</td>
<td width="147">
<p>Encoder-only</p>
</td>
<td width="367">
<pre>sum(
   nearest_key_embedding,
   coorinate_embedding)
</pre>
</td>
<td width="113">
<p>0.654</p>
</td>
</tr>

</tbody>
</table>


From the table above, it can be seen that all existing approaches use similar swipe point embeddings based on the embedding of the nearest key, and other options have not been explored.

At the same time, a hypothesis arises that including information about all keys on the keyboard in the embeddings can mitigate the noise inherent to this task and more accurately reflect the user's interaction with the keyboard. Thus, the research presented in this repository mainly focuses on swipe point representations and their effect on metrics.

## Method

### Model

The model is an encoder-decoder transformer with the following hyperparameters:

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

All experiments utilize this exact model, with the primary difference between the experiments being in the swipe-dot-embedding-layer (embeddings that are encoder input).

Encoder input is a sequence of `swipe point embeddings`. They are described in a dedicated section.

Decoder input is a sequence of trainable embeddings (with positional encoding) of tokens extracted from the target word. In yhis research tokens include all alphabet characters and special tokens (`<sos>`, `<eos>`, `<unk>`, `<pad>`), however the bpe tokens or wordpiece tokens are suitable as well.

The positional encoding is the same as in "Attention is all you need": it's a fixed embedding based on harmonic oscilations.

> [!NOTE]  
> In my research, keyboard key embeddings used in the encoder and character embeddings used in the decoder are different entities.

### Other Models

* There were experiments where the first transformer encoder layer can input a sequence with elements of a dimension different from other encoder layers. I used this kind of custom transformer in Yandex Cup 23, but I don't see the difference in performance or parameter economy as substantial.
* There was an experiment with a larger model: it seems like there is great potential, but it's too expensive to train for me.


### Swipe Point Embeddings


Swipe point embeddings (encoder input) are formed by combining two types of features (though one of them can be omitted): 
1) Features that regard a point as a part of a trajectory
    * Examples: `x`, `y` coordinates; time since beginning of the swipe; $\frac{dx}{dt}$, $\frac{dy}{dt}$; etc.
2) Features that regard a point as a location on a keyboard
    * Examples: a trainable embedding of the nearest keyboard key; a vector of distances between the swipe point and every keyboard key center; etc.

This concept of swipe point embeddings holds true for both: methods presented here and those found in the literature.

> [!NOTE]  
> "SPE" in the sections below stands for Swipe Point mbedding.


#### SPE that uses the nearest key embedding (My Nearest SPE)

This method is the same as indiswipe method but uses second derivatives alongside with other features.

The computational graph of a `swipe point embedding` is shown in the image below.

![Here should be an image of encoder_input_sequence_element](./REAME_materials/encoder_input_sequence_element.png)

The $\frac{dx}{dt}$, $\frac{dy}{dt}$, $\frac{d^2x}{dt^2}$, $\frac{d^2y}{dt^2}$ derivatives are calculated using the finite difference method.


#### SPE that uses a weighted sum of all key embedding (My Weighted SPE)

This is a new method invented in this research that is not found in the literature: all the papers use only the nearest keyboard key embedding when constructing a swipe point embedding.

It is similar to `My nearest SPE` described above, but instead of `the nearest keyboard key embedding` a `weighted sum of all keyboard key embeddings` is used.

$$ embedding = \sum_{key}f(d_{key}) \cdot embedding_{key}$$


Where $d_{key}$ is the distance between between the swipe point and the $key$ center, $f(d_{key})$ is a function that given a distance returns the $key$ weight and $embedding_{key}$ is the $key$'s embedding.

The hyperparameter of the method is the choice of the weighting function $f$. It is assumed that the argument of the function $f$ is the distance from the point to the center of the key, expressed in half-diagonals of the keys. The range of acceptable values is the interval from 0 to 1.

It should be noted that this method is almost the same as `My nearest SPE` if $f$ is a threshold function that returns 1 if $d_{key}$ is less than half the diagonal of the key, and 0 otherwise.


Since the swipes are noisy by their nature and their trajectory often doesn't cross the target keys but always passes near them, the idea arises to take into account all the keys (or in other words, to replace the threshold function with a smooth one).


The weighting function used in this work, along with its graph, is presented below. It is a reflected, right-shifted sigmoid with a sharpened transition from 1 to 0.

$$f(x) = \frac{1}{1+e^{1.8 \cdot (x - 2.2)}}$$


![weights_function](https://github.com/user-attachments/assets/ee43a8bf-cbb1-4885-8a2f-78c2ea034d4f)


In the graph below, the z-axis shows the weight of key `p` (highlighted in yellow) for each point on the keyboard. For other keys, this surface will be the same, just centered at a different point. For a clear view of the keyboard there is a separate image without the surface.



![weights_viz](https://github.com/user-attachments/assets/d2c2505e-91c8-4c33-8bcc-85c386441628)
![3d_keyboard_cut](https://github.com/user-attachments/assets/1ca86f05-b707-41d8-8815-f47fda1f911a)

## Extra Info

Some extra info can be found [solution_description.md](solution_description.md) file (in Russian and may be outdated) and in my [master's thesis]() (in Russian)

**TODO: add link to the thesis**

## Swipe MMR Metric

$
𝑆𝑤𝑖𝑝𝑒 𝑀𝑀𝑅 = 𝐼[cand_1==target]+ 𝐼[cand_2== target]⋅0.1 + 𝐼[cand_3== target]⋅0.09 + 𝐼[cand_4== target]⋅0.08  
$


* 𝑰 is an indicator functoin (returns 1 if the condition in square brackets is met, otherwise returns 0)
* cand_i is the 𝑖-th word candidate list element

All word candidates must be unique. The duplicates are excluded when the metric is calculated


## Results


During inference models are used with a variation of beam search that masks logits corresponding to impossible tokens-continuations given a generated prefix.

Thus the most important metric of the model performance is Swipe MMR when the model is used with "beamsearch with masking". 


The graph below shows that the method proposed in this work demonstrates higher values for both Accuracy and Swipe MMR.



**TODO: Add color descriptions {color -> (used features, paper links)}**




![beamsearch_metrics](https://github.com/user-attachments/assets/cddb1290-8886-4eb4-9366-f072e191e3fc)


The table below demonstrates the best metric values from the graph above for each method. It shows that the developed SPE delivers higher quality than the best SPE used in articles. Specifically, the increase in Swipe MMR is 0.59%, and the increase in accuracy is 0.61%.


Features Type | Swipe MMR | Accuracy | Swipe MMR Epoch | Accuracy Epoch | Max considered epoch  
-------------- | -------- | --------- | -------------- | -------------- | ----
Weighted features (OURS)  | **0.8915**  | **0.8855** | 58 | 58 | 67
Nearest features (OURS)  | 0.8884  | 0.8822 | 33 | 33 | 67
Indiswipe_features  | 0.8863  | 0.8801 | 31 | 31 | 67
Google_2015_features  | 0.8804  | 0.8737 | 53 | 53 | 57
Phrase_swipe_features  | 0.8712  | 0.8645 | 55 | 55 | 64



Features Type | Swipe MMR | Accuracy 
--------------|-----------|------------
Weighted features (OURS) | 0.8915 | 0.8855
Indiswipe_features | 0.8863 | 0.8801
**Δ** | **0.59%** | **0.61%**







Greedy decoding word level accuracy (train set) | CE loss (train set)
:-------------------------:|:-------------------------:
![acc_greedy_TRAIN](https://github.com/user-attachments/assets/b7eda630-b007-442b-b34d-825bb0cd80c4)  |  ![celoss_TRAIN](https://github.com/user-attachments/assets/d801022d-a454-4916-8102-84ec0e228446)

Greedy decoding word level accuracy (validation set) | CE loss (validation set)
:-------------------------:|:-------------------------:
![acc_greedy_val](https://github.com/user-attachments/assets/30dee9b6-a55e-4760-a9fb-9c3ecab4fa79) | ![celoss_val](https://github.com/user-attachments/assets/0f161cdd-8dd9-44bc-b93f-ca6504ea7956)

token level accuracy (validation set) | token level f1-score (validation set)
:-------------------------:|:-------------------------:
![acc_token_val](https://github.com/user-attachments/assets/4ff2a463-1ac8-475c-8ec9-187e821ed229) | ![f1_token_val](https://github.com/user-attachments/assets/5925c80f-4d46-4758-9a56-828f2f7cfe3e)


> [!NOTE]  
> The sudden metrics improvement for `my_nearest_features` and `indiswipe_features` is due to a decrease in learning rate. The losses on the validation set of these models did not change over 20 validations, which led to the ReduceLROnPlateau scheduler cutting the learning rate in half. For other models, ReduceLROnPlateau hasn't taken any actions yet.


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
> `train.ipynb` drains RAM when using `n_workers` > 0 in Dataloader. This can result in up to `dataset_size * n_workers` extra gigabytes of RAM usage. This is a known issue (see [here](https://github.com/pytorch/pytorch/issues/13246)) that happens when a dataset uses a list to store data. Although `torch.cuda.empty_cache()` can be used as a workaround, it doesn't seem to work with pytorch lightning. It appears I didn't commit this workaround, but you can adapt train.ipynb from [before-lightning branch](https://github.com/proshian/neuroswipe/tree/before-lightning) by adding ```torch.cuda.empty_cache()``` after each epoch to to avoid the issue. When training in a kaggle notebook, the issue is not a problem since a kaggle session comes with 30 Gb of RAM.  


## Prediction

[word_generation_demo.ipynb](src/word_generation_demo.ipynb) serves as an example on how to predict via a trained model.

[predict_v2.py](src/predict_v2.py) is used to obtain word candidates for a whole dataset and pickle them

> [!WARNING]  
> If the decoding algorithm in `predict_v2.py` script utilizes a vocabulary for masking (`use_vocab_for_generation: true` in the config), it is necessary to disable multiprocessing by passing the command-line argument `--num-workers 0` to the script. Otherwise, the prediction will take a long time.




## Yandex cup 2023 submission reproduction instructions
Yandex cup 2023 submission reprodction instructions are [here: submission_reproduciton_instrucitons.md](submission_reproduciton_instrucitons.md)




## For future me
See [refactoring plan](./Refactoring_plan.md)