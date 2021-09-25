from functools import lru_cache
import torch

from fewgen.util import all_to, tensor_hashkey
from cachetools import cached, LRUCache
from threading import Lock


def extend_batch_input(inputs, description):
  input_ids, att_mask = inputs
  batch_size = input_ids.shape[0]
  
  ex_ids = torch.tensor(description.prompt + description.ids).long()
  ex_ids = ex_ids.unsqueeze(0).repeat(batch_size, 1)
  ex_mask = torch.ones_like(ex_ids)

  new_input_ids = torch.cat([input_ids, ex_ids], dim=1)
  new_att_mask = torch.cat([att_mask, ex_mask], dim=1)

  return new_input_ids.long(), new_att_mask.long()  


def extend_input(input_, descriptions):
  input_ids, att_mask = input_
  batch_size = len(descriptions)
  
  if len(input_ids.shape) == 1:
    input_ids = input_ids.unsqueeze(0)
    att_mask = att_mask.unsqueeze(0)
    
  input_ids = input_ids.repeat(batch_size, 1)
  att_mask = att_mask.repeat(batch_size, 1)
  
#   max_desc_len = max(map(len, descriptions))
  max_desc_len = max(map(lambda d: len(d.prompt) + len(d.ids), descriptions))
  
  ex_ids, ex_mask = [], []
  for desc in descriptions:   
    desc_ids, desc_mask = desc.to_tensor(prompt=True, pad_to=max_desc_len)
    ex_ids.append(desc_ids)
    ex_mask.append(desc_mask)
  ex_ids = torch.stack(ex_ids, dim=0).long()
  ex_mask = torch.stack(ex_mask, dim=0).long()

  new_input_ids = torch.cat([input_ids, ex_ids], dim=1)
  new_att_mask = torch.cat([att_mask, ex_mask], dim=1)

  return new_input_ids.long(), new_att_mask.long()


generation_cache = LRUCache(maxsize=64)
generation_lock = Lock()

def clear_generation_cache():
  with generation_lock:
    generation_cache.clear()

@cached(cache=generation_cache, key=tensor_hashkey, lock=generation_lock)
@torch.no_grad()
def get_next_probs(model, inputs):
  input_ids, att_mask = inputs
  batch_size = len(input_ids)
  device = model.device
  
  if att_mask.shape[1] > 1 and att_mask[:,-1].all():
    past_ids = input_ids[:, :-1]
    past_mask = att_mask[:, :-1]
    _, past = get_next_probs(model, (past_ids, past_mask))
#     print('using_past', past_ids.shape, tensor_hashkey(model, (past_ids, past_mask)))
  else:
    past = None
  
  prepared_inputs = model.prepare_inputs_for_generation(
    input_ids.to(device), attention_mask=att_mask.to(device), past=all_to(past, device))
    
  outputs = model(**prepared_inputs, return_dict=True)
  logits = outputs['logits'].cpu()
  new_past = all_to(outputs['past_key_values'], device)

  if len(logits.shape) == 2:  # no time dimension (eg. sequence classification model)
    next_logits = logits[:, :]
  elif past is None:
    last_non_masked_idx = torch.sum(att_mask, dim=1) - 1
    next_logits = logits[torch.arange(0, batch_size), last_non_masked_idx]
  else:
    next_logits = logits[:, -1]

  next_probs = next_logits.softmax(dim=-1)  
  return next_probs, new_past


def clear_perplexity_cache():
  with perplexity_lock:
    perplexity_cache.clear()
  return log


@torch.no_grad()
def get_embedding(model, inputs, mode='last_emb'):
  device = model.device
  input_ids, att_mask = inputs
  batch_size = len(input_ids)
  
  prepared_inputs = model.prepare_inputs_for_generation(
    input_ids.to(device), attention_mask=att_mask.to(device)
  )
  
  outputs = model(**prepared_inputs, output_hidden_states=True, return_dict=True)
  embeddings = outputs['hidden_states'][-1]
  
  if 'avg' in mode:
    att_mask = att_mask.to(device)
    masked_embs = embeddings * att_mask.unsqueeze(-1)
    avg_embs = masked_embs.sum(dim=1) / att_mask.sum(dim=1).unsqueeze(-1)
    return avg_embs.cpu()
    
  elif 'last' in mode:
    if 'position_ids' in prepared_inputs:
      last_indixes = prepared_inputs['position_ids'].argmax(dim=1)  # picks first index if duplicate max occurs
    else:
      last_indixes = att_mask.sum(dim=1) - 1  # picks first index if duplicate max occurs
    last_embs = embeddings[torch.arange(batch_size).to(device), last_indixes.to(device)]
    return last_embs.cpu()
    
  else:
    raise ValueError(f'invalid mode: {mode} expcted one of `last_emb`, `avg_emb`')
  


# perplexity_cache = LRUCache(maxsize=128)
# perplexity_lock = Lock()

# @cached(cache=perplexity_cache, key=tensor_hashkey, lock=perplexity_lock)
@torch.no_grad()
def compute_ppl(model, inputs, labels, reduction='none'):
  device = model.device
  input_ids, att_mask = inputs
  
  prepared_inputs = model.prepare_inputs_for_generation(
    input_ids.to(device), attention_mask=att_mask.to(device)
  )
  
  outputs = model(**prepared_inputs, return_dict=True)
  lm_logits = outputs['logits']

  shift_logits = lm_logits[..., :-1, :].contiguous()
  shift_labels = labels[..., 1:].contiguous().to(device)

  loss_fn = torch.nn.CrossEntropyLoss(reduction=reduction).to(device)

  lm_loss = loss_fn(shift_logits.swapaxes(1,2), shift_labels)
  return lm_loss.cpu()


def compute_batch_ppl_change(model, inputs, description):
  ids, mask = inputs
  batch_size = len(ids)
  max_input_len = ids.shape[1]
  prompt_len = len(description.prompt)
  desc_len = len(description.ids)

  desc_ids, desc_mask = description.to_tensor()
  desc_labels = torch.tensor([-100]*prompt_len + description.ids).long()
  pri_ppl = compute_ppl(model, (desc_ids.unsqueeze(0), desc_mask.unsqueeze(0)),
                        desc_labels.unsqueeze(0))

  pri_ppl = pri_ppl.sum(dim=1) / desc_len

  full_ids, full_mask = extend_batch_input(inputs, description)
  full_labels = full_ids.clone()
  full_labels[full_mask==0] = -100
  full_labels[:, :max_input_len+prompt_len] = -100
  post_ppl = compute_ppl(model, (full_ids, full_mask), full_labels)

  post_ppl = post_ppl.sum(dim=1) / desc_len
  return (pri_ppl - post_ppl).exp()


def compute_ppl_changes(model, input_, descriptions):
  ids, mask = input_
  batch_size = len(descriptions)
  max_input_len = ids.shape[1]

  empty_input = torch.tensor([]).long(), torch.tensor([]).long()
  desc_ids, desc_mask = extend_input(empty_input, descriptions)
  desc_labels = desc_ids.clone()
  desc_labels[desc_mask==0] = -100
  for i, d in enumerate(descriptions):
    prompt_len = len(d.prompt)
    desc_labels[i, :prompt_len] = -100
  
  pri_ppl = compute_ppl(model, (desc_ids, desc_mask), desc_labels)
  for i, d in enumerate(descriptions):
    prompt_len = len(d.prompt)
    desc_mask[i, :prompt_len] = 0
  
  pri_ppl = pri_ppl.sum(dim=1) / desc_mask.sum(dim=1)
  
  full_ids, full_mask = extend_input(input_, descriptions)
  full_labels = full_ids.clone()
  full_labels[full_mask==0] = -100
  for i, d in enumerate(descriptions):
    prompt_len = len(d.prompt)
    full_labels[i, :max_input_len+prompt_len] = -100
  
  post_ppl = compute_ppl(model, (full_ids, full_mask), full_labels)
  for i, d in enumerate(descriptions):
    prompt_len = len(d.prompt)
    full_mask[i, :max_input_len+prompt_len] = 0
    
  post_ppl = post_ppl.sum(dim=1) / full_mask.sum(dim=1)
#   print(pri_ppl, post_ppl)
  return (pri_ppl - post_ppl).exp()
   
