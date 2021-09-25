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

from fewgen.util import pad_lists


def finetune_lm(model, tokenizer, dataset, batch_size=4, epochs=5, save_dir='tuned_model', fewshot=True):
  
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
                                    do_train=True, do_eval=True, load_best_model_at_end=not fewshot,  # set load_best_model_at_end=False for real fewshot experiment
                                    per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                                    gradient_accumulation_steps=16//batch_size,
                                    evaluation_strategy='epoch', logging_strategy='epoch',
                                    num_train_epochs=epochs,
                                   )
  
  
  # data_collator = DataCollatorWithPadding(tokenizer, padding='longest')
  callbacks = [] if fewshot else [EarlyStoppingCallback(early_stopping_patience=1)]
  
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
  eval_results = trainer.evaluate()
  # trainer.save_model()  # Saves the tokenizer too for easy upload
  
  return eval_results


def finetune_clf_lm(model, tokenizer, dataset, batch_size=4, epochs=50, save_dir='tuned_model', fewshot=True):
    
  def tokenize_examples(examples):
    return tokenizer(examples['text'], padding='longest', truncation=True)
  
  tokenized_dataset = dataset.map(tokenize_examples, batched=True)
  
  steps_per_epoch = len(tokenized_dataset['train']) // 16
  
  training_args = TrainingArguments(output_dir=save_dir, overwrite_output_dir=True, 
                                    do_train=True, do_eval=True, load_best_model_at_end=not fewshot,
                                    per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                                    evaluation_strategy='steps', gradient_accumulation_steps=16//batch_size,
                                    num_train_epochs=epochs, eval_steps=steps_per_epoch*5, logging_steps=steps_per_epoch*5,
                                   )
  callbacks = [] if fewshot else [EarlyStoppingCallback(early_stopping_patience=1)]
  
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
  # trainer.save_model()  # Saves the tokenizer too for easy upload
  
  return eval_results

