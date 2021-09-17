import os

os.environ['hugging_cache_dir'] = '/home/mohsen/thesis/hugging_cache'

import torch
from transformers import AutoTokenizer, AutoModel, GPT2LMHeadModel
from transformers.modeling_utils import PreTrainedModel
from datasets import load_from_disk
from datasets import load_dataset as origin_load_dataset

from collections import OrderedDict
from cachetools.keys import hashkey
from cachetools import cached, LRUCache

from datasets.utils.logging import set_verbosity_error

set_verbosity_error()


def tensor_hashkey(*args):
  keys = []
  for arg in args:
    if torch.is_tensor(arg):
      keys.append(arg.sum().item())
    elif isinstance(arg, PreTrainedModel):
      keys.append(arg.config._name_or_path)
    elif isinstance(arg, (tuple, list)):
      keys.append(tensor_hashkey(*arg))
    else:
      keys.append(hashkey(arg)[0])
  
  return tuple(keys)


def tokenize_texts(texts, tokenizer):
  tokenized = tokenizer(texts, return_tensors='pt', add_special_tokens=True, padding='longest')
  return (tokenized['input_ids'], tokenized['attention_mask'])

  
def all_to(inputs, device):
  if torch.is_tensor(inputs):
    return inputs.to(device)
  if isinstance(inputs, tuple):
    return tuple([all_to(i, device) for i in inputs])
  if isinstance(inputs, list):
    return [all_to(i, device) for i in inputs]
  if isinstance(inputs, dict):
    return {k:all_to(v, device) for k, v in inputs.items()}


# @cached(LRUCache(maxsize=8))
def load_model(model_name, device='cuda', tokenizer_only=False):
    
  cache_root = os.environ.get('hugging_cache_dir')
  cache_available = cache_root is not None and len(cache_root) > 0
  
  if cache_available:
    tokenizer_path = os.path.join(cache_root, 'tokenizer', model_name)
    model_path = os.path.join(cache_root, 'model', model_name)
  else:
    tokenizer_path = model_name
    model_path = model_name
    

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=cache_available)
  if tokenizer_only:
    return tokenizer
  
  if 'gpt' in model_name:
    tokenizer.pad_token = tokenizer.bos_token
    tokenizer.mask_token = tokenizer.bos_token

    model = GPT2LMHeadModel.from_pretrained(model_path, pad_token_id=tokenizer.bos_token_id,
                                            local_files_only=cache_available)
  else:
      model = AutoModel.from_pretrained(model_path, local_files_only=cache_available)
  return tokenizer, model.to(device)


def load_dataset(dataset_name, subtask=None):
    
  cache_root = os.environ.get('hugging_cache_dir')
  cache_available = cache_root is not None and len(cache_root) > 0
  
  if cache_available:
    dataset_dir = os.path.join(dataset_name, subtask) if subtask else dataset_name
    dataset = load_from_disk(os.path.join(cache_root, 'dataset', dataset_dir))
  elif subtask is None:
    dataset = origin_load_dataset(*dataset_name.split('/'))
  else:
    dataset = origin_load_dataset(dataset_name, subtask)
    
  return dataset


def load_nli(model_name, device='cpu'):
  tokenizer, model = load_model(model_name, device)
  
  return ZeroShotClassificationPipeline(
    model=model, tokenizer=tokenizer,
    device=-1 if device == 'cpu' else 0
  )