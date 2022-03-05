# FewGen
**Few**shot Text Classification using **Gen**erative Language Models (eg. `GPT2`)

Check out the [demo](https://colab.research.google.com/github/tabasy/fewgen/blob/main/demo.ipynb) in Google Colab!

## Define FewGen Classifier
First consider a single-input text classification task (our default task is `SST2`). To create a `FewgenClassifier`, we need:
* a **language model** along with its tokenizer
* some **descriptions** for each class of your task

For the language model part, only variants of `GPT2` are tested, but any model instantiated by `transformers.AutoModelForCausalLM` is expected to work.

```python
from fewgen.util import load_model

tokenizer, language_model = load_model('gpt2-medium', device='cuda')
clear_output()
```

The descriptions are defined as below. Each description has two parts which are separated with a ` / `:
> `" All in all, the movie was a / terrible failure"`

The first part acts as a **prompt** and then comes the **answer** part. 

For training and inference, the descriptions are scored by calculating perplexity change for the answer part (before and after prepending our given input text).
> *All in all, the movie was a* ***terrible failure***

> I would not recommend it to anyone. *All in all, the movie was a* ***terrible failure***

Now you are free to define your own descriptions:

```python
descriptions = {
  # the keys are class labels
  'neg': [
    " All in all, the movie was a / terrible failure",
    " I would give it / zero of 10",
    " All in all, the screenplay is / poorly written",
    " I think the actors / played awful"
  ],
  'pos': [
    " All in all, the movie was a / masterpiece",
    " I would give it / 10 stars of 10",
    " All in all, the screenplay is / very well written",
    " I think the actors / should win the award"
  ]
}
```
Now we are ready to instantiate a `FewgenClassifier`:

```python
from fewgen.classifier import FewgenClassifier

classifier = FewgenClassifier(descriptions=descriptions, 
                              language_model=language_model,
                              tokenizer=tokenizer)
```

## Training

To train our classifier, we need a dataset. It is supposed to be a (huggingface) `datasets.Dataset` instance, including two fields:
* the `text` field, our input text
* the `label` field, with values matching one of our defined `descriptions.keys()`

Without finetuning language model, the training means fitting a linear model on description scores (as high-level features).

```python
classifier.train(trainset, finetune_lm=False)
classifier.test(testset)
```

And we have the option to finetune language model and get better results.

```python
classifier.train(trainset, finetune_lm=True, finetune_args=dict(epochs=3))
```

## Inference
Model inference could not be easier than this:
```python
text = "I would not recommend it to anyone."

pred = classifier.classify(text)[0]        # 'neg'
probs = classifier.predict_proba(text)[0]  # [89.1, 10.9]
```
## Interpretaion
One of the benefits of prompt-based methods (most of them), is their **interpretability**. As the prompt and the answers are in natural language and seem humanly understandable, we expect to understand how the model decides (in some degrees)!

We can check how prepending the input text to a description, changes the probability of answer words (and th conditional perplexity). 
For this part, check out the [demo](https://colab.research.google.com/github/tabasy/fewgen/blob/main/demo.ipynb).
