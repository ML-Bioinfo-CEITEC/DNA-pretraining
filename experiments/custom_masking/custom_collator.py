from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import DataCollatorForLanguageModeling
import torch

import numpy as np

class SubsequentCollator(DataCollatorForLanguageModeling):
    def __init__(self, mask_fully=False, **kwargs):
        super().__init__(**kwargs)
        self.mask_fully=mask_fully
    
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample consecutive tokens
        probability_matrix = self.get_probability_matrix_consecutive(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% masking 
        if(self.mask_fully):
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            return inputs, labels
            
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def get_probability_matrix_consecutive(self, labels_shape, mlm_probability):
        batch_size, seq_len = labels_shape
        masksize = int(seq_len*mlm_probability)
        result = torch.zeros((batch_size, seq_len))
        ind = np.random.randint(0,seq_len-masksize, batch_size) #TODO 512 included = add -1?
        mask_indicies = torch.stack([torch.arange(b_ind, b_ind+masksize) for b_ind in ind])
        result.scatter_(-1, mask_indicies, 1)
        return result
    

class WideCollator(DataCollatorForLanguageModeling):
    def __init__(self, area, mask_fully=False ,**kwargs):
        super().__init__(**kwargs)
        self.mask_fully=mask_fully
        self.area = area
    
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample consecutive tokens
        probability_matrix = self.get_probability_matrix_wide(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% masking 
        if(self.mask_fully):
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
            return inputs, labels
            
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def get_probability_matrix_wide(self, labels_shape, mlm_probability):
        batch_size, seq_len = labels_shape
        masksize = int(seq_len*mlm_probability)
        masked_k = self.area
        
        #maskpercentage = overall tokens VS single tokens + dont count neighbours
        num_of_areas = masksize//masked_k
        # num_of_areas = masksize
        # print(num_of_areas, 'noa')
        result = torch.zeros((batch_size, seq_len))
        #TODO returns deterministic values!!! Fix seed?
        # print(torch.randperm(seq_len-masked_k+1)[:num_of_areas])
        #TODO indicies are not exclusively masked (may overlap = randperm from seq_len/kmer_size and then upscale?
        ind = torch.stack([torch.randperm(seq_len-masked_k+1)[:num_of_areas] for _ in range(batch_size)] )
        ind = torch.cat([ind+k for k in range(0, masked_k)],1)
        ind_wide = torch.unique(ind, dim=1)
        result.scatter_(-1, ind_wide, 1)
        # print(torch.count_nonzero(result[0]))
        # print(torch.count_nonzero(result[1]))
        # print(result[0])
        # print(result[1])
        
        return result