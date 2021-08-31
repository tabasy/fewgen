from datasets import load_dataset, list_datasets, concatenate_datasets


def balance_dataset(dataset, label_name='label', ex_per_class=None):
  
  if ex_per_class < -1:
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
    if ex_per_class is None or ex_per_class <= 0 or len(subset) < ex_per_class:
      ex_per_class = len(subset)

  subsets = [subset.select(torch.arange(ex_per_class)) for subset in subsets]
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


def prepare_dataset(dataset, label_name, ex_per_class, base_model_name,
                    rules, templates, batch_size=32, convert_label=False):
  balanced = balance_dataset(dataset, label_name, ex_per_class)
  reformatted = reformat_dataset(balanced, rules, templates)
  
  text_fields = {k: k for k in templates}
  tokenized = tokenize_dataset(reformatted, text_fields, base_model_name, batch_size)
  
  input_fields = [k+'_input_ids' for k in templates]
  mask_fields = [k+'_attention_mask' for k in templates]
  tokenized.set_format(type='pt', columns=input_fields+mask_fields+['label_int' if convert_label else label_name])
  
  clear_output()
  
  return tokenized

def tokenize_texts(texts, tokenizer):
  tokenized = tokenizer(texts, return_tensors='pt', add_special_tokens=True, padding='longest')
  return (tokenized['input_ids'], tokenized['attention_mask'])
