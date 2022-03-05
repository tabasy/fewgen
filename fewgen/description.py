#@title description class
import re, json
import torch
from functools import total_ordering
from anytree import NodeMixin
from difflib import SequenceMatcher
from scipy.stats import spearmanr


@total_ordering
class Description(NodeMixin):
  punc_re = re.compile(r'[,:;!?\'"`()�_—-]+')
#   eps = 2e-3

  @staticmethod
  def set_hparams(punc_re=None):
#     if eps is not None:
#       Description.eps = eps
    if punc_re is not None:
      Description.punc_re = re.compile(punc_re)
    
  @staticmethod
  def scorer(probs_pos, probs_neg):
    if probs_neg is None or len(probs_neg) == 0:
      return probs_pos
    return probs_pos / probs_neg

  @staticmethod
  def reducer(scores):
    log_scores = torch.log(scores)
    return torch.exp(torch.mean(log_scores))
      
  @staticmethod
  def from_text(text=None, prompt=None, tokenizer=None, full=None):
    if full:
      prompt, text = full.split('/')
      prompt = ' ' + prompt.strip()
      text =  ' ' + text.strip()

    return Description(tokenizer.encode(text, add_special_tokens=False),
                       tokenizer=tokenizer,
                       prompt=prompt)

  def __init__(self, ids=None, pos_probs=None, neg_probs=None, parent=None, tokenizer=None, prompt=None):
    self.ids = ids or []
    self.pos_probs = pos_probs or [] 
    self.neg_probs = neg_probs or [] 
    self.score = None
    self.penalty = 1.0
    
    self.parent = parent
    self.tokenizer = tokenizer or parent.tokenizer
    
    if prompt is None:
      self.prompt = parent.prompt
    elif isinstance(prompt, str):
      self.prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
    else:
      self.prompt = prompt
      
  
  def set_prompt(self, prompt):
    self.prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
    
  def migrate(self, new_tokenizer):
    
    if new_tokenizer == self.tokenizer:
      return self
    
    desc = self.tokenizer.decode(self.ids)
    prompt = self.tokenizer.decode(self.prompt)
        
    new_desc = Description.from_text(desc, prompt, new_tokenizer)
    
    return new_desc
    
  def generate(self, pos_probs, neg_probs=None, n=8):
    scores = Description.scorer(pos_probs, neg_probs)
    top_k = scores.topk(n).indices

    new_gen = []
    for idx in top_k:
      if neg_probs is None:
        new_neg_probs = []
      else:
        new_neg_probs = self.neg_probs + [neg_probs[idx].item()]
      
      description = Description(
          ids=self.ids+[idx.item()],
          pos_probs=self.pos_probs+[pos_probs[idx].item()],
          neg_probs=new_neg_probs,
          tokenizer=self.tokenizer, prompt=self.prompt
          )
      if description.validate():
        new_gen.append(description)
      else:
        del description
    self.children = new_gen[:n]
    return self.children

  def delete(self):
    self.parent = None

  def get_score(self):
    if len(self.ids) == 0:
      return -float('inf')
    if self.score is None:
      scores = Description.scorer(torch.tensor(self.pos_probs), torch.tensor(self.neg_probs))
      self.score = Description.reducer(scores)
    return self.score * self.penalty

  def penalize(self, others, factor=0.9):
    matcher = SequenceMatcher(b=self.ids)
    for other in others:
      matcher.set_seq1(other.ids)
      lcs = matcher.find_longest_match(0, len(other), 0, len(self)).size
      if lcs > 0:
        self.penalty *= factor ** (lcs - 0.5)
  
  def absolve(self):
    self.penalty = 1.0

  def get_text(self, inputs=None, prompt=False, prompt_only=False):
    if inputs is None:
      if prompt_only:
        return self.tokenizer.decode(self.prompt)
      elif prompt:
        return self.tokenizer.decode(self.prompt) + ' /' + self.tokenizer.decode(self.ids)
      else:
        return self.tokenizer.decode(self.ids)
    else:
      return self.tokenizer.batch_decode(self.extend_inputs(inputs), skip_special_tokens=True)

  def validate(self):
    text = self.get_text()
    if len(self.punc_re.findall(text)) > 0:
      return False
    if '\n' in text:
      return False
    return True
  
  def to_tensor(self, prompt=True, pad_to=None):
    pad_token_id = self.tokenizer.pad_token_id
    all_ids = self.prompt + self.ids
    pad_len = pad_to - len(all_ids) if pad_to else 0
    
    padding_ids = [pad_token_id] * pad_len
    padding_mask = [0] * pad_len

    ids = torch.tensor(all_ids + padding_ids).long()
    mask = torch.tensor([1] * len(all_ids) + padding_mask).long()
    
    return ids, mask

  def __str__(self):
    return self.get_text(prompt=True)

  def __lt__(self, other):
    return self.get_score() < other.get_score()

  def __eq__(self, other):
    return self.ids == other.ids

  def __len__(self):
    return len(self.ids)

  def __repr__(self):
    if self.penalty == 1.0:
      return f'Description(text="{self.get_text()}", score={self.get_score():.4f})'
    else:
      return f'Description(text="{self.get_text()}", score={self.get_score():.4f}, penalty={self.penalty:.4f})'

    
def save_descriptions(c2d, path, names, params=None):
  
  jdata = {'descriptions': {}}
  
  if params:
    jdata['params'] = params
  
  for c, ds in c2d.items():
    name = names[c]
    jdata['descriptions'][name] = [d.get_text(prompt=True, prompt_only=False) for d in ds]
  
  with open(path, 'w', encoding='utf8') as outf:
    json.dump(jdata, outf, ensure_ascii=False, indent=2)

    
def load_descriptions(path, names, num_desc=0):
  
  with open(path, encoding='utf8') as inf:
    jdesc = json.load(inf)
  
  c2d = {}
  
  for name, ds in jdesc['descriptions'].items():
    c = names.index(name)

    if num_desc > 0:
      c2d[c] = ds[:num_desc]
    else:
      c2d[c] = ds
      
  return c2d

    
class Prompt(Description):
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    
    self.discrimination_score = None
    
  def compute_discrimination_score(self, next_pos_probs, next_neg_probs):
    sp_corr = spearmanr(next_pos_probs.numpy(), next_neg_probs.numpy())
    self.discrimination_score = - sp_corr.correlation
    
  def generate(self, probs, n=8):
    top_k = probs.topk(n).indices

    new_gen = []
    for idx in top_k:
      child = Prompt(
          ids=self.ids+[idx.item()],
          pos_probs=self.pos_probs+[probs[idx].item()],
          tokenizer=self.tokenizer, prompt=self.prompt
          )
      if child.validate():
        new_gen.append(child)
      else:
        del child
    self.children = new_gen[:n]
    return self.children
  
  def __repr__(self):
    if self.discrimination_score is None:
      return f'Prompt(text="{self.get_text()}", score={self.get_score():.4f})'
    else:
      return f'Prompt(text="{self.get_text()}", score={self.get_score():.4f}, disc_score={self.discrimination_score:.4f})'
