import os
import numpy as np
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.data.data_collator import DataCollatorWithPadding
from datasets import load_dataset, load_from_disk, load_metric
from datasets import Dataset, DatasetDict

from fewgen.util import pad_lists


def generate_finetuning_data(results, reverse_train=False, reverse_test=False, save_dir=None):   # use ppl change to select better descriptions for each example
  trainset, testset = results['trainset'], results['testset']
  class_descriptions = results['class_descriptions']

  train_data = {'text': [], 'prompt': [], 'description': [], 'full': []}
  test_data = {'text': [], 'prompt': [], 'description': [], 'full': []}
  
  lebel_set = set(trainset['label'])
  
  for text, label in zip(trainset['text'], trainset['label']):
    if reverse_train:
      label = choice(list(lebel_set - {label}))
    for desc in class_descriptions[label]:
      train_data['text'].append(text)
      train_data['prompt'].append(desc.get_text(prompt_only=True))
      train_data['description'].append(desc.get_text(prompt=False))
      train_data['full'].append(f'{text.strip()} {desc.get_text(prompt=True)}')
  
  for text, label in zip(testset['text'], testset['label']):
    if reverse_test:
      label = choice(list(lebel_set - {label}))
    for desc in class_descriptions[label]:
      test_data['text'].append(text)
      test_data['prompt'].append(desc.get_text(prompt_only=True))
      test_data['description'].append(desc.get_text(prompt=False))
      test_data['full'].append(f'{text.strip()} {desc.get_text(prompt=True)}')
      
  datasets = DatasetDict({
    'train': Dataset.from_dict(train_data),
    'validation': Dataset.from_dict(test_data),
    'test': Dataset.from_dict(test_data), 
    }
  )
  
  if save_dir:
    os.makedirs(save_dir, exist_ok=True)
    datasets.save_to_disk(save_dir)
      
  return datasets


def finetune_lm(model, tokenizer, dataset, batch_size=4, epochs=2, steps=-1, save_dir='DO_NOT_SAVE',
                early_stopping_threshold=0.01):
#   epochs = int(os.environ.get('epoch', 2))
  
  def tokenize_example(example):
    tokenized = {}

    text = tokenizer(example['text'])
    prompt = tokenizer(example['prompt'])
    desc = tokenizer(example['description'])

    tokenized['input_ids'] = text['input_ids'] + prompt['input_ids'] + desc['input_ids']
    tokenized['attention_mask'] = text['attention_mask'] + prompt['attention_mask'] + desc['attention_mask']
    tokenized['labels'] = [-100] * len(text['input_ids'] + prompt['input_ids']) + desc['input_ids']

    return tokenized

  def pad_examples(examples, pad_token_id=0):
    padded = {}

    padded['input_ids'] = pad_lists(examples['input_ids'], pad_token_id)
    padded['attention_mask'] = pad_lists(examples['attention_mask'], 0)
    padded['labels'] = pad_lists(examples['labels'], -100)

    return padded

  if isinstance(dataset, str):
    dataset = load_from_disk(dataset)
  
  tokenized_dataset = dataset.map(tokenize_example, batched=False)
  padded_dataset = tokenized_dataset.map(pad_examples,
                                         batched=True, batch_size=None,
                                         fn_kwargs={'pad_token_id': tokenizer.pad_token_id})
  
  training_args = TrainingArguments(output_dir=save_dir, overwrite_output_dir=True,
                                    do_train=True, do_eval=True, load_best_model_at_end=False,
                                    per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                                    gradient_accumulation_steps=16//batch_size,
                                    max_steps=steps, num_train_epochs=epochs,
                                    logging_first_step=True, #logging_steps=steps//5, eval_steps=steps//5,
                                    evaluation_strategy='epoch', logging_strategy='epoch', save_strategy='no', 
                                   )
  
  # data_collator = DataCollatorWithPadding(tokenizer, padding='longest')
  callbacks = []#[EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=early_stopping_threshold)]
  
  trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=padded_dataset['train'],
        eval_dataset=padded_dataset['validation'],
        tokenizer=tokenizer,
        callbacks=callbacks,
        data_collator=default_data_collator,
    )
    
  trainer.train()
  if save_dir != 'DO_NOT_SAVE':
    trainer.save_model()  # Saves the tokenizer too for easy upload
  
  eval_results = trainer.evaluate()
  return eval_results


def finetune_clf_lm(model, tokenizer, dataset, batch_size=4, epochs=20, save_dir='DO_NOT_SAVE',
                    early_stopping_threshold=0.001):
#   epochs = int(os.environ.get('epoch', 2))
    
  def tokenize_examples(examples):
    return tokenizer(examples['text'], padding='longest', truncation=True)
  
  tokenized_dataset = dataset.map(tokenize_examples, batched=True)
  steps_per_epoch = max(1, len(tokenized_dataset['train']) // 16)
  
  training_args = TrainingArguments(output_dir=save_dir, overwrite_output_dir=True,
                                    do_train=True, do_eval=True, load_best_model_at_end=False,
                                    per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                                    gradient_accumulation_steps=16//batch_size,
                                    logging_first_step=True, num_train_epochs=epochs,
                                    evaluation_strategy='steps', logging_strategy='steps', save_strategy='no', 
                                    save_steps=steps_per_epoch*5, eval_steps=steps_per_epoch*5, logging_steps=steps_per_epoch*5,
                                   )
    
  callbacks = []#[EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=early_stopping_threshold)]
  
  metric = load_metric('accuracy')

  def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
  
  trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        tokenizer=tokenizer,
        callbacks=callbacks,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator, # default_data_collator
    )
  
  trainer.train()
  eval_results = trainer.evaluate()
  
  if save_dir != 'DO_NOT_SAVE':
    trainer.save_model()  # Saves the tokenizer too for easy upload
  
  return eval_results