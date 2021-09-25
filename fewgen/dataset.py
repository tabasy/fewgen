import numpy as np
from datasets import concatenate_datasets

from fewgen.util import load_dataset


TEXT_FIELD_NAMES = ['text', 'sentence', 'review', 'comment']


def unify_text_fields(dataset, max_length=32):
  
  def unify_text_field(example):
    
    new_example = example.copy()
    
    for name in TEXT_FIELD_NAMES:
      if name in example:
        new_example['text'] = example[name]
        break
    return new_example
  
  def filter_by_length(example):
    return example['text'].count(' ') < max_length

  unified = dataset.map(unify_text_field)
  return unified.filter(filter_by_length)


def balance_dataset(dataset, label_name='label', ex_per_class=-1):
  """ 
    ex_per_class:
    None or 0: do nothing
    -1 : max balanced
  n > 0: normal behavior
  """
  if ex_per_class in [0, None]:
    return dataset

  classes = sorted(dataset.unique(label_name))

  def make_balanced_index(example, index):
    class_num = classes.index(example[label_name])
    example['label_int'] = class_num
    example['balanced_index'] = len(classes) * (index) + class_num
    return example

  subsets = []

  for i, label in enumerate(classes):
    subset = dataset.select(np.where(np.array(dataset[label_name]) == label)[0])
    subsets.append(subset)
    if ex_per_class < 0 or len(subset) < ex_per_class:
      ex_per_class = len(subset)

  subsets = [subset.select(np.arange(ex_per_class)) for subset in subsets]
  subsets = [subset.map(make_balanced_index, with_indices=True) for subset in subsets]

  return concatenate_datasets(subsets).sort('balanced_index')

  
def reformat_dataset(dataset, rules, templates):

  def apply_template(example, template):

    text_input = template

    for tag, repl in rules.items():
      if isinstance(repl, str) and repl in example:
        repl = example[repl]
      if callable(repl):
        repl = repl(example)
      text_input = text_input.replace(tag, str(repl))

    return text_input
     
  def apply_templates(example):
    new_example = deepcopy(example)

    for name, template in templates.items():
      new_example[name] = apply_template(example, template) 
    
    return new_example

  return dataset.map(apply_templates)


def append_prompt(dataset, prompt):
  
  def add_prompt(example):
      example['text'] = example['text'] + prompt
      return example
    
  return dataset.map(add_prompt)


def tokenize_dataset(dataset, text_fields, base_model_name='roberta-base',
                     batch_size=32):
  
  tokenizer = AutoTokenizer.from_pretrained(base_model_name)
  if 'gpt' in base_model_name:
    tokenizer.pad_token = tokenizer.eos_token

  def tokenize_batch(batch_example):

    for inp, out in text_fields.items():
      tokenized = tokenizer(batch_example[inp], padding='longest')
      batch_example[out + '_input_ids'] = tokenized['input_ids']
      batch_example[out + '_attention_mask'] = tokenized['attention_mask']

    return batch_example

  return dataset.map(tokenize_batch, batched=True, batch_size=batch_size)


def get_dataset_label_names(dataset):
  if hasattr(dataset.info.features['label'], 'names'):
    return dataset.info.features['label'].names
  return list(map(str, sorted(np.unique(dataset['label']).tolist())))


def prepare_dataset(dataset_name, subset_name=None, shuffle=False, shuffle_seed=0,
                    train_ex_per_class=16, test_ex_per_class=None,
                    test_split_name='validation', prompt=None):
  
  dataset = load_dataset(dataset_name, subset_name)
  
  
  trainset = dataset['train']
  testset = dataset[test_split_name]

  trainset = unify_text_fields(trainset, max_length=64)
  testset = unify_text_fields(testset, max_length=64)
  
  if shuffle:
    trainset = trainset.shuffle(shuffle_seed)
    
  trainset = balance_dataset(trainset.flatten_indices(), label_name='label', ex_per_class=train_ex_per_class)
  testset = balance_dataset(testset.flatten_indices(), label_name='label', ex_per_class=test_ex_per_class)
  
  
  if prompt is not None and len(prompt.strip()) > 0:
    prompt = ' ' + prompt.strip()
    trainset = append_prompt(trainset, prompt)
    testset = append_prompt(testset, prompt)
  
  return trainset, testset
