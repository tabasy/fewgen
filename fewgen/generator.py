import torch
from anytree import LevelOrderGroupIter
from tqdm.auto import tqdm

from fewgen.description import Description, Prompt
from fewgen.inference import *
from fewgen.util import *


class DiverseDescriptionGenerator:
  def __init__(self, model_name=None, tokenizer=None, model=None, device='cuda'):
    
    if tokenizer is None or model is None:
       tokenizer, model = load_model(model_name, device)

    self.tokenizer = tokenizer
    self.lm = model
        
    self.prompt = None
    self.beam_size = 16
    self.num_beam_groups = 4
    self.min_len = 2
    self.max_len = 5
    self.satisfaction = None
    self.diversity_factor = 0.95
    self.stop_if_satisfied = True
    self.keep_longest = False
    self.group_best_only = True
    self.quantile_value = 0.5
    self.log = False


  def set_hparams(self, **hparams):
    self.__dict__.update(hparams)

  def batch_generate(self, inputs, neg_inputs=None):
    is_single_input = len(inputs[0]) == 1
    descriptions = []
    roots = [Description(tokenizer=self.tokenizer, prompt=self.prompt) for i in range(self.num_beam_groups)]
    group_iters = [LevelOrderGroupIter(root) for root in roots]

    for group_levels in tqdm(zip(*group_iters), total=self.max_len, disable=is_single_input):

      group_beams = []

      for glevel in group_levels:
        
        for gbeam in group_beams:  # just previous beams
          for desc in glevel:
            desc.penalize(gbeam, factor=self.diversity_factor)

        sorted_glevel = sorted(glevel, reverse=True)
        new_gbeam = sorted_glevel[:self.beam_size]
        group_beams.append(new_gbeam)

        for desc in sorted_glevel[self.beam_size:]:
          desc.delete()
          
        desc_to_generate = []

        for i, desc in enumerate(new_gbeam):
          if self.log and len(desc) > 0:
            print(repr(desc))

          satisfied = self.satisfaction and self.satisfaction(desc)
          last_step = len(desc) == self.max_len
          is_group_best = last_step and i == 0
          keep_free = not self.group_best_only

          keep = self.group_best_only and is_group_best
          keep = keep or (satisfied and keep_free)
          keep = keep or (last_step and self.keep_longest and keep_free)
          skip = last_step or (satisfied and self.stop_if_satisfied)

          if keep:
            descriptions.append(desc)                     

          if skip:
            continue
            
          desc_to_generate.append(desc)
          
        if self.log:
          print('-'*50)
          
        if len(desc_to_generate) == 0:
            continue
        
        if is_single_input:  # single inputs -> batch over descriptions
          pos_probs = get_next_probs(self.lm, extend_input(inputs, desc_to_generate))[0]
          if neg_inputs is None:
            neg_probs = [None] * len(desc_to_generate) 
          else:
             neg_probs = get_next_probs(self.lm, extend_input(neg_inputs, desc_to_generate))[0]
          for desc, pos_pr, neg_pr in zip(desc_to_generate, pos_probs, neg_probs):
            desc.generate(pos_pr, neg_pr, self.beam_size)
        
        else:   # batch inputs
          for desc in desc_to_generate:
            pos_probs = get_next_probs(self.lm, extend_batch_input(inputs, desc))[0].quantile(q=self.quantile_value, dim=0)
            neg_probs = None if neg_inputs is None else get_next_probs(self.lm, extend_batch_input(neg_inputs, desc))[0].quantile(q=self.quantile_value, dim=0)
            desc.generate(pos_probs, neg_probs, self.beam_size)

      if self.log:
        print('='*50)
        
    clear_generation_cache()

    for desc in descriptions:
      desc.absolve()   # clear penalties

    descriptions = sorted(descriptions, reverse=True)[:self.beam_size]
    return descriptions

  def generate_class_descriptions(self, texts, labels):

    ids, masks = tokenize_texts(texts, self.tokenizer)
    labels = torch.tensor(labels)
    descriptions = {}

    for cls_ in torch.unique(labels):
      pos_inputs = ids[labels==cls_], masks[labels==cls_]
      all_neg_inputs = ids[labels!=cls_], masks[labels!=cls_]
      neg_indices = torch.randperm(len(all_neg_inputs[0]))[:len(pos_inputs[0])]
      neg_inputs = all_neg_inputs[0][neg_indices], all_neg_inputs[1][neg_indices]

      descriptions[cls_.item()] = self.batch_generate(pos_inputs, neg_inputs)
      
    return descriptions
  
  def generate_example_descriptions(self, text):
    if self.prompt is None or self.prompt == '':
      inputs = tokenize_texts([text.strip()], self.tokenizer)
      neg_inputs = None
    else:
      inputs = tokenize_texts([text], self.tokenizer)
      neg_inputs = tokenize_texts([''], self.tokenizer)
    
    descriptions = self.batch_generate(inputs, neg_inputs)
    return descriptions
  
class DiversePromptGenerator:
  def __init__(self, model_name=None, tokenizer=None, model=None, device='cuda'):
    
    if tokenizer is None or model is None:
       tokenizer, model = load_model(model_name, device)

    self.tokenizer = tokenizer
    self.lm = model
        
    self.prompt = ''
    self.beam_size = 16
    self.num_beam_groups = 4
    self.min_len = 2
    self.max_len = 5
    self.diversity_factor = 0.95
    self.min_discrimination_score = -0.98
    self.log = False

  def set_hparams(self, **hparams):
    self.__dict__.update(hparams)

  def batch_generate(self, inputs, neg_inputs=None):
    prompts = []
    roots = [Prompt(tokenizer=self.tokenizer, prompt=self.prompt) for i in range(self.num_beam_groups)]
    group_iters = [LevelOrderGroupIter(root) for root in roots]

    for group_levels in tqdm(zip(*group_iters), total=self.max_len):

      group_beams = []

      for glevel in group_levels:
        
        for gbeam in group_beams:  # just previous beams
          for desc in glevel:
            desc.penalize(gbeam, factor=self.diversity_factor)

        sorted_glevel = sorted(glevel, reverse=True)
        new_gbeam = sorted_glevel[:self.beam_size]
        group_beams.append(new_gbeam)

        for desc in sorted_glevel[self.beam_size:]:
          desc.delete()
          
        for i, cand in enumerate(new_gbeam):
            
          pos_probs = get_next_probs(self.lm, extend_batch_input(inputs, cand))[0].quantile(q=self.quantile_value, dim=0)
          neg_probs = get_next_probs(self.lm, extend_batch_input(neg_inputs, cand))[0].quantile(q=self.quantile_value, dim=0)
          all_probs = (pos_probs + neg_probs) / 2.0
          top_indices = all_probs.argsort(descending=True)[:50]
          cand.compute_discrimination_score(pos_probs[top_indices], neg_probs[top_indices])
          
          if len(cand) < self.max_len:
            cand.generate(all_probs, n=self.beam_size)
          
          if cand.discrimination_score >= self.min_discrimination_score:
            prompts.append(cand)    
            
          if self.log and len(cand) > 0:
            print(repr(cand), cand.discrimination_score)
          
        if self.log:
          print('-'*50)

      if self.log:
        print('='*50)

    for prompt in prompts:
      prompt.absolve()   # clear penalties

    prompts = sorted(prompts, reverse=True, key=lambda p: p.discrimination_score)[:self.beam_size]
    return prompts

  def generate_discriminative_prompts(self, texts, labels):

    ids, masks = tokenize_texts(texts, self.tokenizer)
    labels = torch.tensor(labels)
    prompts = {}

    for cls_ in torch.unique(labels):
      pos_inputs = ids[labels==cls_], masks[labels==cls_]
      all_neg_inputs = ids[labels!=cls_], masks[labels!=cls_]
      neg_indices = torch.randperm(len(all_neg_inputs[0]))[:len(pos_inputs[0])]
      neg_inputs = all_neg_inputs[0][neg_indices], all_neg_inputs[1][neg_indices]

      prompts[cls_.item()] = self.batch_generate(pos_inputs, neg_inputs)
      
      return prompts