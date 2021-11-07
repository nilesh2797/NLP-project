#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys, math, random, time
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer


def ToD(batch, device):
    if isinstance(batch, torch.Tensor):
        batch = batch.to(device)
    if isinstance(batch, Dict):
        for outkey in batch:
            if isinstance(batch[outkey], torch.Tensor):
                batch[outkey] = batch[outkey].to(device)
            if isinstance(batch[outkey], Dict):
                for inkey in batch[outkey]:
                    if isinstance(batch[outkey][inkey], torch.Tensor):
                        batch[outkey][inkey] = batch[outkey][inkey].to(device)
    return batch

def np_relu(x):
    return np.maximum(0, x)

def get_index_values(spmat, row_index, add_one=False):
    start = spmat.indptr[row_index]; end = spmat.indptr[row_index+1]
    row_data = spmat.data[start:end]
    row_indices = spmat.indices[start:end]
    
    if(add_one):
        row_indices = row_indices + 1

    return row_indices, row_data

def csr_to_bow_tensor(spmat):
    return {'inputs': torch.LongTensor(spmat.indices),
            'offsets': torch.LongTensor(spmat.indptr),
            'per_sample_weights': torch.Tensor(spmat.data)}

def csr_to_pad_tensor(spmat, pad):
    maxlen = spmat.getnnz(1).max()
    ret = {'inds': torch.full((spmat.shape[0], maxlen), pad).long().flatten(),
           'vals': torch.zeros(spmat.shape[0], maxlen).flatten()}
    ptrs = []
    for i in range(spmat.shape[0]):
        ptrs.append(torch.arange(i*maxlen, i*maxlen + spmat.indptr[i+1] - spmat.indptr[i]))
    ptrs = torch.cat(ptrs)
    ret['inds'][ptrs] = torch.LongTensor(spmat.indices)
    ret['inds'] = ret['inds'].reshape((spmat.shape[0], maxlen))
    ret['vals'][ptrs] = torch.Tensor(spmat.data)
    ret['vals'] = ret['vals'].reshape((spmat.shape[0], maxlen))
    return ret

def bert_fts_batch_to_tensor(input_ids, attention_mask):
    maxlen = attention_mask.sum(axis=1).max()
    return {'input_ids': torch.LongTensor(input_ids[:, :maxlen]), 
            'attention_mask': torch.LongTensor(attention_mask[:, :maxlen])}
    
def bow_fts_batch_to_tensor(batch):
    xlen = sum([len(b['inds']) for b in batch])
    ret = {'inputs': torch.zeros(xlen).long(), 
           'offsets': torch.zeros(len(batch)+1).long(),
           'per_sample_weights': torch.zeros(xlen)}
    offset = 0
    for i, b in enumerate(batch):
        new_offset = offset+len(b['inds'])
        ret['inputs'][offset:new_offset] = torch.Tensor(b['inds']).long()
        ret['per_sample_weights'][offset:new_offset] = torch.Tensor(b['vals'])
        ret['offsets'][i+1] = new_offset            
        offset = new_offset
    return ret

def get_cross_fts(batch_data, tokenizer):
    ft_len = batch_data['xfts']['input_ids'].shape[1]+1+batch_data['yfts']['input_ids'].shape[1]
    num_shorty = batch_data['shorty']['vals'].shape[1]
    device = batch_data['xfts']['input_ids'].device

    cross_fts = {'input_ids': torch.full((batch_data['batch_size'], num_shorty, ft_len), tokenizer.pad_token_id, device=device).long(),
                 'attention_mask': torch.zeros(batch_data['batch_size'], num_shorty, ft_len, device=device).long()}

    for i in range(batch_data['batch_size']):
        ilen = batch_data['xfts']['attention_mask'][i].sum()
        totlen = ilen+batch_data['yfts']['input_ids'].shape[1]
        cross_fts['input_ids'][i, :, :totlen] = torch.hstack((batch_data['xfts']['input_ids'][i, :ilen].reshape(1, -1).repeat_interleave(num_shorty, dim=0),
                                                  torch.full((num_shorty, 1), tokenizer.sep_token_id, device=device),
                                                  batch_data['yfts']['input_ids'][batch_data['shorty']['batch-inds'][i].flatten()][:, 1:]))
        cross_fts['attention_mask'][i, :, :totlen] = torch.hstack((batch_data['xfts']['attention_mask'][i, :ilen].reshape(1, -1).repeat_interleave(num_shorty, dim=0),
                                                  torch.full((num_shorty, 1), 1, device=device),
                                                  batch_data['yfts']['attention_mask'][batch_data['shorty']['batch-inds'][i].flatten()][:, 1:]))
    cross_fts['input_ids'] = cross_fts['input_ids'].reshape(-1, ft_len)
    cross_fts['attention_mask'] = cross_fts['attention_mask'].reshape(-1, ft_len)
    batch_data['cross-fts'] = cross_fts

class FixedDataset(torch.utils.data.Dataset):

    def __init__(self, point_features, labels, label_features=None, shorty=None):
        self.point_features = point_features
        self.label_features = label_features
        self.labels = labels
        self.shorty = shorty

        print("------ Some stats about the dataset ------")
        print("Shape of X_Xf       : ", self.point_features.shape)
        print("Shape of X_Y        : ",  self.labels.shape, end='\n\n')
        print("Avg. lbls per point : %.2f"%(np.average(np.array(self.labels.sum(axis=1)))))
        print("Avg. fts per point  : %.2f"%(np.average(self.point_features.astype(np.bool).sum(axis=1))))
        print("------------------------------------------")

        self.num_Y = self.labels.shape[1]
        self.num_Xf = self.point_features.shape[1]

    def __getitem__(self, index):
        ret = {'index': index, 'xfts': None, 'y': None, 'shorty': None}
        ret['xfts'] = self.get_fts(index, 'point')
        temp = get_index_values(self.labels, index)
        ret['y'] = {'inds': temp[0], 'vals': temp[1]}
        
        if not self.shorty is None:
            temp = get_index_values(self.shorty, index)
            ret['shorty'] = {'inds': temp[0], 'vals': temp[1]}
        return ret
    
    def get_fts(self, index, source='label'):
        if isinstance(self.point_features, sp.csr_matrix):
            if isinstance(index, int) or isinstance(index, np.int32):
                if source == 'label':
                    temp = get_index_values(self.label_features, index)
                else:
                    temp = get_index_values(self.point_features, index)
                return {'inds': temp[0], 
                        'vals': temp[1]}
            else:
                if source == 'label':
                    return csr_to_bow_tensor(self.label_features[index])
                else:
                    return csr_to_bow_tensor(self.point_features[index])
        else:
            if isinstance(index, int) or isinstance(index, np.int32):
                if source == 'label':
                    return self.label_features[index]
                else:
                    return self.point_features[index]
            else:
                if source == 'label':
                    return torch.Tensor(self.label_features[index])
                else:
                    return torch.Tensor(self.point_features[index])

    @property
    def num_instances(self):
        return self.point_features.shape[0]

    @property
    def num_labels(self):
        return self.labels.shape[1]
    
    def __len__(self):
        return self.point_features.shape[0]

class PreTokBertDataset(torch.utils.data.Dataset):
    def __init__(self, tokenization_folder, X_Y, max_len, shorty=None, doc_type='trn', iter_mode='pointwise'):
        self.max_len = max_len
        self.iter_mode = iter_mode
        self.labels = X_Y
        self.shorty = shorty
        
        if not os.path.exists(tokenization_folder):
            print(f'Pre-Tokenized folder ({tokenization_folder}) not found')
            print(f'Help: python CreateTokenizedFiles.py --data-dir Datasets/<dataset> --max-length {max_len}')
            sys.exit()
        
        self.X_ii = np.memmap(f"{tokenization_folder}/{doc_type}_doc_input_ids.dat", 
                             mode='r', shape=(X_Y.shape[0], max_len), dtype=np.int64)
        self.X_am = np.memmap(f"{tokenization_folder}/{doc_type}_doc_attention_mask.dat", 
                              mode='r', shape=(X_Y.shape[0], max_len), dtype=np.int64)
        
        self.Y_ii = np.memmap(f"{tokenization_folder}/lbl_input_ids.dat", 
                              mode='r', shape=(X_Y.shape[1], max_len), dtype=np.int64)
        self.Y_am = np.memmap(f"{tokenization_folder}/lbl_attention_mask.dat", 
                              mode='r', shape=(X_Y.shape[1], max_len), dtype=np.int64)
            
    def __getitem__(self, index):
        ret = {'index': index}
        return ret
    
    def get_fts(self, indices, source='point'):
        if source == 'point':
            return bert_fts_batch_to_tensor(self.X_ii[indices], self.X_am[indices])
        if source == 'label':
            return bert_fts_batch_to_tensor(self.Y_ii[indices], self.Y_am[indices])
   
    def __len__(self):
        return self.labels.shape[0]

# ## Collators

# In[ ]:


class XCCollator():
    def __init__(self, numy, dataset, yfull=False):
        self.numy = numy
        self.yfull = yfull
        self.dataset = dataset
    
    def __call__(self, batch):
        batch_size = len(batch)
        ids = np.array([b['index'] for b in batch])
        
        batch_data = {'batch_size': torch.LongTensor([batch_size]),
                      'numy': torch.LongTensor([self.numy]),
                      'y': csr_to_pad_tensor(self.dataset.labels[ids], self.numy),
                      'ids': torch.LongTensor(ids),
                      'xfts': self.dataset.get_fts(ids, 'point')
                     }
            
        if self.dataset.shorty is not None:
            batch_data['shorty'] = csr_to_pad_tensor(self.dataset.shorty[ids], self.numy)
            
        if self.yfull:
            batch_data['yfull'] = torch.zeros(batch_size, self.numy+1).scatter_(1, batch_data['y']['inds'], batch_data['y']['vals'])[:, :-1]
                
        return batch_data


# In[ ]:


class SiameseTrainCollator():
    def __init__(self, dataset, yfull=False, _type='batch-rand', num_neg=1):
        self.numy = dataset.labels.shape[1]
        self.yfull = yfull
        self.dataset = dataset
        self._type = _type
        self.num_neg = num_neg
        self.mask = torch.zeros(self.numy+1).long()
        self.merge_fts_func = dataset.merge_fts_func
    
    def __call__(self, batch):
        batch_size = len(batch)
        ids = np.array([b['index'] for b in batch])
        
        batch_data = {'batch_size': batch_size,
                      'numy': self.numy,
                      'shorty': None,
                      'y': csr_to_pad_tensor(self.dataset.labels[ids], self.numy),
                      'yfull': None,
                      'ids': torch.Tensor([b['index'] for b in batch]).long(),
                      'xfts': self.dataset.get_fts(ids, 'point')
                     }
        
        batch_y = None
        
        if self._type == 'shorty':
            batch_data['shorty'] = csr_to_pad_tensor(self.dataset.shorty[ids], self.numy)
            batch_y_inds = torch.multinomial(batch_data['y']['vals'].double(), 1)
            batch_pos_y = torch.gather(batch_data['y']['inds'], 1, batch_y_inds).squeeze()
            batch_y = torch.LongTensor(np.union1d(batch_pos_y, batch_data['shorty']['inds']))
            
            self.mask[batch_y] = torch.arange(batch_y.shape[0])
            batch_data['pos-inds'] = self.mask[batch_pos_y].reshape(-1, 1)
            batch_data['shorty']['batch-inds'] = self.mask[batch_data['shorty']['inds'].flatten()].reshape(batch_data['shorty']['inds'].shape)
            self.mask[batch_y] = 0
            
            batch_data['targets'] = torch.zeros((batch_size, batch_y.shape[0]))
            for i in range(batch_size):
                self.mask[batch_data['y']['inds'][i]] = True
                batch_data['targets'][i][self.mask[batch_y].bool()] = 1.0
                self.mask[batch_data['y']['inds'][i]] = False
        
        elif self._type == 'batch-rand':
            batch_y_inds = torch.multinomial(batch_data['y']['vals'].double(), 1)
            batch_y = torch.gather(batch_data['y']['inds'], 1, batch_y_inds).squeeze()
            batch_data['pos-inds'] = torch.arange(batch_size).reshape(-1, 1)
            batch_data['targets'] = torch.zeros((batch_size, batch_y.shape[0]))
            for i in range(batch_size):
                self.mask[batch_data['y']['inds'][i]] = True
                batch_data['targets'][i][self.mask[batch_y].bool()] = 1.0
                self.mask[batch_data['y']['inds'][i]] = False
                
        if batch_y is not None:
            batch_data['batch_y'] = batch_y
            batch_data['yfts'] = self.dataset.get_fts(batch_y.numpy(), 'label')
            
        if self.yfull:
            batch_data['yfull'] = torch.zeros(batch_size, self.numy+1).scatter_(1, batch_data['y']['inds'], batch_data['y']['vals'])[:, :-1]
                
        return batch_data


# In[ ]:


## example usage
# trn_dataset = BertDataset(trnX, Y, trn_X_Y, shorty=trn_shorty, maxsize=128, data_root_dir=OUT_DIR)

# collator = SiameseTrainCollator(params.numy, trn_dataset)
# trn_loader = torch.utils.data.DataLoader(
#     trn_dataset,
#     batch_size=32,
#     num_workers=1,
#     collate_fn=collator,
#     shuffle=True,
#     pin_memory=True)


# ## Model Classes

# In[ ]:


from collections import OrderedDict
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm.autonotebook import tqdm, trange
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable
import logging
logger = logging.getLogger(__name__)

import transformers
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, RobertaModel

class Params:
    def __init__(self):
        pass


# ### Blocks

# In[ ]:


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        pass
    def forward(self, input):
        return F.normalize(input)

class Linear(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(Linear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = Parameter(torch.Tensor(self.output_size, self.input_size))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        return F.linear(input, self.weight, self.bias.view(-1))

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

class SparseLinear(nn.Module):
    def __init__(self, hidden_size, num_labels, device_embeddings, padding_idx=None, bias=True):
        super(SparseLinear, self).__init__()
        self.padding_idx = padding_idx
        self.device_embeddings = device_embeddings
        self.input_size = hidden_size
        self.output_size = num_labels
        self.sp_weight = Parameter(torch.Tensor(self.output_size, self.input_size))
        if bias:
            self.sp_bias = Parameter(torch.Tensor(self.output_size, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.sparse = True  # Required for optimizer

    def forward(self, embed, shortlist):
        embed = embed.to(self.device_embeddings)
        shortlist = shortlist.to(self.device_embeddings)
        short_weights = F.embedding(shortlist,
                                    self.sp_weight,
                                    sparse=self.sparse,
                                    padding_idx=self.padding_idx)
        out = torch.matmul(embed.unsqueeze(1), short_weights.permute(0, 2, 1))
        if self.sp_bias is not None:
            short_bias = F.embedding(shortlist,
                                     self.sp_bias,
                                     sparse=self.sparse,
                                     padding_idx=self.padding_idx)
            out = out + short_bias.permute(0, 2, 1)
        return out.squeeze()

    def reset_parameters(self):
        print("Sparse linear is getting initialized.")
        stdv = 1. / math.sqrt(self.sp_weight.size(1))
        self.sp_weight.data.uniform_(-stdv, stdv)
        if self.sp_bias is not None:
            self.sp_bias.data.uniform_(-stdv, stdv)
        if self.padding_idx is not None:
            self.sp_weight.data[self.padding_idx].fill_(0)

    def move_to_devices(self):
        super().to(self.device_embeddings)

class Residual(nn.Module):
    def __init__(self, input_size, output_size, dropout, init='eye'):
        super(Residual, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.init = init
        self.dropout = dropout
        self.padding_size = self.output_size - self.input_size
        print("Using residual block with constrained spectral norm.")
        self.hidden_layer = nn.Sequential(
                    nn.utils.spectral_norm(nn.Linear(self.input_size, self.output_size)),
                    nn.ReLU(),
                    nn.Dropout(self.dropout))
        self.initialize(self.init)

    def forward(self, embed):
        temp = F.pad(embed, (0, self.padding_size), 'constant', 0)
        embed = self.hidden_layer(embed) + temp
        return embed

    def initialize(self, init_type):
        if init_type == 'random':
            nn.init.xavier_uniform_(
                self.hidden_layer[0].weight,
                gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(self.hidden_layer[0].bias, 0.0)
        else:
            print("Using eye to initialize!")
            nn.init.eye_(self.hidden_layer[0].weight)
            nn.init.constant_(self.hidden_layer[0].bias, 0.0)


# ### Generic Model

# In[ ]:


class GenericTrainer(nn.Module):
    def __init__(self, 
                 train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
                 evaluator = None,
                 name: str = 'default',
                 out_dir: str = None,
                 device: str = None, 
                 use_amp: bool = False,
                 scheduler: str = 'WarmupLinear',
                 warmup_steps: int = 10000,
                 dense_optimizer_class: Type[Optimizer] = torch.optim.Adam,
                 dense_optimizer_params : Dict[str, object]= {'lr': 0.025},
                 sp_optimizer_class: Type[Optimizer] = torch.optim.SparseAdam,
                 sp_optimizer_params : Dict[str, object]= {'lr': 0.025},
                 tf_optimizer_class: Type[Optimizer] = transformers.AdamW,
                 tf_optimizer_params : Dict[str, object]= {'lr': 2e-5, 'eps': 1e-6, 'weight_decay': 0.01},):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.scaler = torch.cuda.amp.GradScaler(init_scale=2**12)
        self._target_device = torch.device(device)
        self.use_amp = use_amp
        self.out_dir = out_dir
        self.name = name
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"{name} on device: {device}")
        
        self.evaluator = evaluator
        self.dataloaders = [dataloader for dataloader, _ in train_objectives]
        self.loss_models = [loss for _, loss in train_objectives]
        
        # Prepare optimizers
        self.scheduler_name = scheduler
        self.warmup_steps = warmup_steps
        self.dense_optimizer_class = dense_optimizer_class
        self.dense_optimizer_params = dense_optimizer_params
        self.sp_optimizer_class = sp_optimizer_class
        self.sp_optimizer_params = sp_optimizer_params
        self.tf_optimizer_class = tf_optimizer_class
        self.tf_optimizer_params = tf_optimizer_params
            
    def prepare_optimizers(self, num_train_steps):
        self.optimizers = []
        self.schedulers = []
        
        for loss_model in self.loss_models:
            optimizer_params = {'sp': [], 'dense': [], 'tf': []}
            no_decay_params = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            sparse_params = ['sp_emb.weight', 'sp_weight', 'sp_bias']
            
            for n, p in loss_model.named_parameters():
                if p.requires_grad:
                    if 'transformer' in n: 
                        optimizer_params['tf'].append((n, p))
                    elif any(sp in n for sp in sparse_params):
                        optimizer_params['sp'].append((n, p))
                    else:
                        optimizer_params['dense'].append((n, p))
            
            tf_optimizer_grouped_parameters = [
                {'params': [p for n, p in optimizer_params['tf'] if not any(nd in n for nd in no_decay_params)], 'weight_decay': self.tf_optimizer_params['weight_decay']},
                {'params': [p for n, p in optimizer_params['tf'] if any(nd in n for nd in no_decay_params)], 'weight_decay': 0.0}
            ]
            dense_optimizer_grouped_parameters = [
                {'params': [p for n, p in optimizer_params['dense']], 'weight_decay': self.dense_optimizer_params['weight_decay']},
            ]
            sp_optimizer_grouped_parameters = [
                {'params': [p for n, p in optimizer_params['sp']]},
            ]

            optimizer = []
            if len(dense_optimizer_grouped_parameters[0]['params']) > 0:
                optimizer.append(self.dense_optimizer_class(dense_optimizer_grouped_parameters, **self.dense_optimizer_params))
            if len(tf_optimizer_grouped_parameters[0]['params']) > 0:
                optimizer.append(self.tf_optimizer_class(tf_optimizer_grouped_parameters, **self.tf_optimizer_params))
            if len(sp_optimizer_grouped_parameters[0]['params']) > 0:
                optimizer.append(self.sp_optimizer_class(sp_optimizer_grouped_parameters, **self.sp_optimizer_params))
                
            scheduler_obj = [self._get_scheduler(op, scheduler=self.scheduler_name, warmup_steps=self.warmup_steps, t_total=num_train_steps) for op in optimizer]
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler_obj)
        
    def fit(self,
            epochs: int = 1,
            steps_per_epoch = None,
            evaluation_epochs: int = 5,
            save_best: bool = True,
            max_grad_norm: float = -1,
            callback_func = None
            ):
        
        for loss_model in self.loss_models:
            loss_model.to(self._target_device)
            loss_model.zero_grad()
            loss_model.train()
        
        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in self.dataloaders])
        num_train_steps = int(steps_per_epoch * epochs)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in self.dataloaders]
        num_train_objectives = len(self.loss_models)
        skip_scheduler = False
        
        self.prepare_optimizers(num_train_steps)
        
        for epoch in trange(epochs, desc="Epoch"):
            training_steps = 0
            total_loss = 0
            for loss_model in self.loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05):
                for train_idx in range(num_train_objectives):
                    loss_model = self.loss_models[train_idx]
                    optimizer = self.optimizers[train_idx]
                    scheduler = self.schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        batch_data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(self.dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        batch_data = next(data_iterator)
                    batch_data = self.batch_to_device(batch_data, self._target_device)
                    
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        loss_value = loss_model(batch_data)
                    total_loss += loss_value.item()
                    
                    if self.use_amp: self.scaler.scale(loss_value).backward()
                    else: loss_value.backward()
                            
                    if max_grad_norm > 0: 
                        for op in optimizer: 
                            if self.use_amp: self.scaler.unscale_(op)
                        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                    for op in optimizer: 
                        if self.use_amp: self.scaler.step(op)
                        else: op.step()
                    for op in optimizer: op.zero_grad()

                    if not skip_scheduler:
                        for sch in scheduler: sch.step()
                            
                    if self.use_amp: self.scaler.update()

                training_steps += 1
                global_step += 1
                del batch_data
            
            mean_loss = total_loss/training_steps
            print(f'mean loss after epoch {epoch} : {"%.4E"%(mean_loss)}')
            if epoch%evaluation_epochs == 0 and self.evaluator is not None:
                score = self.evaluator(epoch, mean_loss, self.out_dir, self.name)    
                        
            if callback_func is not None:
                callback_func(epoch, self.dataloaders)
                data_iterators = [iter(dataloader) for dataloader in self.dataloaders]

    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))
            
    def batch_to_device(self, batch, device):
        if isinstance(batch, torch.Tensor):
            batch = batch.to(device)
        if isinstance(batch, Dict):
            for outkey in batch:
                if isinstance(batch[outkey], torch.Tensor):
                    batch[outkey] = batch[outkey].to(device)
                if isinstance(batch[outkey], Dict):
                    for inkey in batch[outkey]:
                        if isinstance(batch[outkey][inkey], torch.Tensor):
                            batch[outkey][inkey] = batch[outkey][inkey].to(device)
        return batch


# In[ ]:


class GenericModel(nn.Sequential):
    def __init__(self, modules: Iterable[nn.Module] = None, name: str = 'generic_model', out_dir: str = None, device: str = None, encoder_offset: int = 1, encoder_normalize: bool = False):
        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        super().__init__(modules)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._target_device = torch.device(device)
        self.encoder_offset = encoder_offset
        self.encoder_normalize = encoder_normalize
        self.out_dir = out_dir
        self.name = name
        os.makedirs(self.out_dir, exist_ok=True)
        
    def encode(self, inp):
        for i, module in enumerate(self):
            if i >= len(self)-self.encoder_offset:
                break
            inp = module(inp)
        if self.encoder_normalize: return F.normalize(inp)
        else: return inp
    
    def get_embs(self, dataset, source='point', bsz=256):
        if source == 'label':
            numx = dataset.labels.shape[1]
        elif source == 'point':
            numx = dataset.labels.shape[0]
            
        embs = [] 
        self.eval()
        with torch.no_grad():
            for ctr in tqdm(range(0, numx, bsz), desc=f"Embedding {source}s"):
                batch_data = dataset.get_fts(np.array(range(ctr, min(numx, ctr+bsz))), source)
                batch_data = self.batch_to_device(batch_data, self._target_device)
                temp_embs = self.encode(batch_data)
                embs.append(temp_embs.detach().cpu().numpy())
                del temp_embs, batch_data
        return np.vstack(embs)
            
    def batch_to_device(self, batch, device):
        if isinstance(batch, torch.Tensor):
            batch = batch.to(device)
        if isinstance(batch, Dict):
            for outkey in batch:
                if isinstance(batch[outkey], torch.Tensor):
                    batch[outkey] = batch[outkey].to(device)
                if isinstance(batch[outkey], Dict):
                    for inkey in batch[outkey]:
                        if isinstance(batch[outkey][inkey], torch.Tensor):
                            batch[outkey][inkey] = batch[outkey][inkey].to(device)
        return batch
    
    def save(self, out_dir: str = None):
        if out_dir is None: out_dir = self.out_dir
        os.makedirs(out_dir, exist_ok=True)
        torch.save(self.state_dict(), f'{out_dir}/{self.name}_state_dict.pt')
    
    def load(self, out_dir: str = None):
        if out_dir is None: out_dir = self.out_dir
        self.load_state_dict(torch.load(f'{out_dir}/{self.name}_state_dict.pt'))


# ## Helper Classes

# ### Input layers

# In[2]:


class BagInputLayer(nn.Module):
    def __init__(self, pretrained_embs=None, shape=None):
        super(BagInputLayer, self).__init__()
        if pretrained_embs is not None:
            self.sp_emb = nn.EmbeddingBag.from_pretrained(pretrained_embs, freeze=False, sparse=True, mode="sum", include_last_offset=True)
        elif shape is not None:
            self.sp_emb = nn.EmbeddingBag(*shape, sparse=True, mode="sum", include_last_offset=True)
        else:
            print('ERROR: both pretrained_embs and shape are None')
            
    def forward(self, data):
        return self.sp_emb( data['inputs'], 
                            data['offsets'], 
                            data['per_sample_weights'])

class STransformerInputLayer(nn.Module):
    def __init__(self, transformer='roberta-base'):
        super(STransformerInputLayer, self).__init__()
        if isinstance(transformer, str):
            self.transformer = SentenceTransformer(transformer)
        else:
            self.transformer = transformer

    def forward(self, data):
        return self.transformer(data)['sentence_embedding']

class TransformerInputLayer(nn.Module):
    def __init__(self, transformer, pooler_type='pooler'):
        super(TransformerInputLayer, self).__init__()
        self.transformer = transformer
        self.pooler = self.create_pooler(pooler_type)

    def forward(self, data):
        return self.pooler(self.transformer(**data), data)
    
    def create_pooler(self, pooler_type: str):
        if pooler_type == 'seq-clf':
            def f(tf_output, batch_data):
                return tf_output.logits
            return f
        elif pooler_type == 'pooler':
            def f(tf_output, batch_data):
                return tf_output['pooler_output']
            return f
        elif pooler_type == 'cls':
            def f(tf_output, batch_data):
                return tf_output['last_hidden_state'][:, 0]
            return f
        elif pooler_type == 'mean':
            def f(tf_output, batch_data):
                last_hidden_state = tf_output['last_hidden_state']
                input_mask_expanded = batch_data['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_hidden_state = torch.sum(last_hidden_state * input_mask_expanded, 1)

                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)

                return sum_hidden_state / sum_mask
            return f
        else:
            print(f'Unknown pooler type encountered: {pooler_type}, using identity pooler instead')
            def f(tf_output, batch_data):
                return tf_output
            return f
            

class CustomRobertaInputLayer(nn.Module):
    def __init__(self, shape):
        super(CustomRobertaInputLayer, self).__init__()
        self.emb = nn.Embedding(*shape, padding_idx=1)

    def forward(self, data):
        token_embs = self.emb(data['input_ids'])
        input_mask_expanded = data['attention_mask'].unsqueeze(-1).expand(token_embs.size()).float()
        sum_token_embs = torch.sum(token_embs * input_mask_expanded, 1)

        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)

        return sum_token_embs / sum_mask


# ### Loss models

# In[ ]:


def _reduce(val, reduction='mean'):
    if reduction == 'mean':
        return val.mean()
    elif reduction == 'sum':
        return val.sum()

class BCELoss(nn.Module):
    def __init__(self, model: GenericModel, reduction='mean'):
        super(BCELoss, self).__init__()
        self.model = model
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, batch_data):
        out = self.model(batch_data['xfts'])
        if batch_data['yfull'] is None:
            batch_data['yfull'] = torch.zeros(batch_data['batch_size'], batch_data['numy']+1, device=batch_data['y']['inds'].device).scatter_(1, batch_data['y']['inds'], batch_data['y']['vals'])[:, :-1]

        loss = self.criterion(out, batch_data['yfull'])
        del batch_data, out
        return loss

class SparseBCELoss(nn.Module):
    def __init__(self, model: GenericModel, reduction='mean'):
        super(SparseBCELoss, self).__init__()
        self.model = model
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, batch_data):
        embs = self.model.encode(batch_data['xfts'])
        out = self.model[-1](embs, batch_data['shorty']['inds'])
        loss = self.criterion(out, batch_data['shorty']['vals'])
        del batch_data, embs, out
        return loss

class SiameseTripletOHNM(nn.Module):
    def __init__(self, model: GenericModel, numy, reduction='mean', margin=0.8, k=3):
        super(SiameseTripletOHNM, self).__init__()
        self.model = model
        self.numy = numy
        self.reduction = reduction
        self.margin = margin
        self.k = k

    def forward(self, batch_data):
        xembs = self.model.encode(batch_data['xfts'])
        yembs = self.model.encode(batch_data['yfts'])
        target = batch_data['targets']
        sim = xembs @ yembs.T
        sim_p = torch.gather(sim, 1, batch_data['pos-inds']).reshape(batch_data['batch_size'], -1)
        neg_sim = torch.where(target < 1e-6, sim, torch.full_like(sim, -100))
        _, indices = torch.topk(neg_sim, largest=True, dim=1, k=self.k)
        sim_n = sim.gather(1, indices)
        loss = torch.max(torch.zeros_like(sim_p), sim_n - sim_p + self.margin)
        mask = torch.where(loss != 0, torch.ones_like(loss), torch.zeros_like(loss))
        prob = torch.softmax(sim_n * mask, dim=1)
        reduced_loss =  _reduce(loss * prob, self.reduction)
        
        del batch_data, xembs, yembs, sim, neg_sim, sim_p, sim_n, mask, prob, loss
        return reduced_loss

class SiameseBCELoss(nn.Module):
    def __init__(self, model: GenericModel, numy, reduction='mean'):
        super(SiameseBCELoss, self).__init__()
        self.model = model
        self.numy = numy
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, batch_data):
        xembs = self.model.encode(batch_data['xfts'])
        yembs = self.model.encode(batch_data['yfts'])
        short_weights = F.embedding(batch_data['shorty']['batch-inds'],
                                    yembs,
                                    padding_idx=None)
        out = torch.matmul(xembs.unsqueeze(1), short_weights.permute(0, 2, 1)).squeeze()
        loss = self.criterion(out, batch_data['shorty']['vals'])
        del batch_data, xembs, yembs, out, short_weights
        return loss

class SiameseCrossEntropyLoss(nn.Module):
    def __init__(self, model: GenericModel, numy, reduction='mean', scale=1):
        super(SiameseCrossEntropyLoss, self).__init__()
        self.model = model
        self.numy = numy
        self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
        self.scale = scale

    def forward(self, batch_data):
        xembs = self.model.encode(batch_data['xfts'])
        yembs = self.model.encode(batch_data['yfts'])
        sim = (xembs @ yembs.T)*self.scale
        labels = torch.multinomial(batch_data['shorty']['vals'].double(), 1).squeeze()
        loss = self.criterion(sim, labels)
        del batch_data, xembs, yembs, sim, labels
        return loss


# In[ ]:


class CrossBCELoss(nn.Module):
    def __init__(self, model: GenericModel, tokenizer, reduction='mean'):
        super(CrossBCELoss, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, batch_data):
        get_cross_fts(batch_data, self.tokenizer)
        out = self.model(batch_data['cross-fts'])
        loss = self.criterion(out, batch_data['shorty']['vals'].reshape(-1, 1))
        del batch_data, out
        return loss


# In[ ]:


class kPCLoss(nn.Module):
    r""" A probabilistic contrastive loss 
    *expects cosine similarity
    * or <w, x> b/w normalized vectors
    Arguments:
    ----------
    reduction: string, optional (default='mean')
        Specifies the reduction to apply to the output:
        * 'none': no reduction will be applied
        * 'mean' or 'sum': mean or sum of loss terms
    pos_weight: float or None, optional (default=None)
        weight of loss with positive target
    """
    def __init__(self, model: GenericModel, reduction='mean', pos_weight=1.0, c=0.9, d=1.5, k=2, apply_softmax=True):
        super(kPCLoss, self).__init__()
        self.model = model
        self.pos_weight = pos_weight
        self.d = d
        self.k = k
        self.c = math.log(c)
        self.scale = 1/d
        self.constant = c/math.exp(d)
        self.apply_softmax = apply_softmax
        
    def forward(self, batch_data):
        xembs = self.model.encode(batch_data['xfts'])
        yembs = self.model.encode(batch_data['yfts'])
        input = (xembs @ yembs.T)
        target = batch_data['targets']
        loss = torch.where(
            target > 0, -self.c + (1-input)*self.d*self.pos_weight,
            -torch.log(1 - torch.exp(self.d*input)*self.constant))
        neg_vals, neg_ind = torch.topk(loss-target*3, k=self.k)
        loss_neg = torch.zeros_like(target)
        if self.apply_softmax:
            neg_probs = torch.softmax(neg_vals, dim=1)
            loss_neg = loss_neg.scatter(1, neg_ind, neg_probs*neg_vals)  #loss.gather(1, indices)
        else:
            loss_neg = loss_neg.scatter(1, neg_ind, neg_vals)
        loss = torch.where(
            target > 0, loss, loss_neg)  
        del xembs, yembs, input, target
        return _reduce(loss)


# ### Predictors

# In[1]:


class FullPredictor():
    def __init__(self, K=100):
        self.K = K

    def __call__(self, model: GenericModel, dataloader: DataLoader):
        datalen = len(dataloader.dataset)
        data = np.zeros((datalen, self.K))
        inds = np.zeros((datalen, self.K)).astype(np.int32)
        indptr = np.arange(0, datalen*self.K+1, self.K)
        ctr = 0; numy = None
        model.eval()
        
        with torch.no_grad():
            for step, batch_data in enumerate(tqdm(dataloader, desc="Evaluating")):
                batch_data = model.batch_to_device(batch_data, model._target_device)
                out = model(batch_data['xfts'])
                if numy is None: numy = out.shape[1]

                bsz = batch_data['batch_size']
                top_data, top_inds = torch.topk(out, self.K)
                data[ctr:ctr+bsz] = top_data.detach().cpu().numpy()
                inds[ctr:ctr+bsz] = top_inds.detach().cpu().numpy()

                ctr += bsz
                del top_data, top_inds, batch_data, out
        torch.cuda.empty_cache() 
        return csr_matrix((data.ravel(), inds.ravel(), indptr), (datalen, numy))

class SparsePredictor():
    def __init__(self, K=100):
        self.K = K

    def __call__(self, model: GenericModel, dataloader: DataLoader):
        datalen = len(dataloader.dataset)
        numy = dataloader.dataset.num_Y
        K = min(self.K, len(dataloader.dataset[0]['shorty']['inds']))
        data = np.zeros((datalen, K))
        inds = np.zeros((datalen, K)).astype(np.int32)
        indptr = np.arange(0, datalen*K+1, K)
        ctr = 0; 
        model.eval()
        
        with torch.no_grad():
            for step, batch_data in enumerate(tqdm(dataloader, desc="Evaluating")):
                batch_data = model.batch_to_device(batch_data, model._target_device)
                embs = model.encode(batch_data['xfts'])
                out = model[-1](embs, batch_data['shorty']['inds'])

                bsz = batch_data['batch_size']
                top_data, top_inds = torch.topk(out, K)
                top_inds = torch.gather(batch_data['shorty']['inds'], 1, top_inds)
                data[ctr:ctr+bsz] = top_data.detach().cpu().numpy()
                inds[ctr:ctr+bsz] = top_inds.detach().cpu().numpy()

                ctr += bsz
                del embs, top_data, top_inds, batch_data, out
        torch.cuda.empty_cache() 
        return csr_matrix((data.ravel(), inds.ravel(), indptr), (datalen, numy))

class SiameseFullPredictor():
    def __init__(self, K=100, bsz=1024):
        self.K = K
        self.bsz = bsz

    def __call__(self, model: GenericModel, dataloader: DataLoader):
        dataset = dataloader.dataset
        
        with torch.no_grad():
            xembs = model.get_embs(dataset, 'point', self.bsz)
            yembs = model.get_embs(dataset, 'label', self.bsz)
            
            torch.cuda.empty_cache()
            es = exact_search({'data': yembs, 'query': xembs, 'K': self.K, 'device': model._target_device})
            return es.getnns_gpu()

class SiameseSparsePredictor():
    def __init__(self, K=100):
        self.K = K

    def __call__(self, model: GenericModel, dataloader: DataLoader):
        datalen = len(dataloader.dataset)
        numy = dataloader.dataset.num_Y
#         K = min(self.K, len(dataloader.dataset[0]['shorty']['inds']))
        K = self.K
        data = np.zeros((datalen, K))
        inds = np.zeros((datalen, K)).astype(np.int32)
        indptr = np.arange(0, datalen*K+1, K)
        ctr = 0; 
        model.eval()
        
        with torch.no_grad():
            for step, batch_data in enumerate(tqdm(dataloader, desc="Evaluating")):
                batch_data = model.batch_to_device(batch_data, model._target_device)
                xembs = model.encode(batch_data['xfts'])
                yembs = model.encode(batch_data['yfts'])
                short_weights = F.embedding(batch_data['shorty']['batch-inds'],
                                            yembs,
                                            padding_idx=None)
                out = torch.matmul(xembs.unsqueeze(1), short_weights.permute(0, 2, 1)).squeeze()

                bsz = batch_data['batch_size']
                top_data, top_inds = torch.topk(out, K)
                top_inds = torch.gather(batch_data['shorty']['inds'], 1, top_inds)
                data[ctr:ctr+bsz] = top_data.detach().cpu().numpy()
                inds[ctr:ctr+bsz] = top_inds.detach().cpu().numpy()

                ctr += bsz
                del xembs, yembs, top_data, top_inds, batch_data, out, short_weights
        torch.cuda.empty_cache() 
        return csr_matrix((data.ravel(), inds.ravel(), indptr), (datalen, numy))


# In[ ]:


class CrossPredictor():
    def __init__(self, K=10):
        self.K = K

    def __call__(self, model: GenericModel, dataloader: DataLoader):
        K = len(dataloader.dataset[0]['shorty']['inds'])
        datalen = len(dataloader.dataset)
        numy = dataloader.dataset.labels.shape[1]
        
        data = np.zeros((datalen, K))
        inds = np.zeros((datalen, K)).astype(np.int32)
        indptr = np.arange(0, datalen*K+1, K)
        ctr = 0
        model.eval()
        
        with torch.no_grad():
            for batch_data in tqdm(dataloader):
                bsz = batch_data['batch_size']
                model.batch_to_device(batch_data, model._target_device)
                get_cross_fts(batch_data, dataloader.dataset.tokenizer)
                
                data[ctr:ctr+bsz] = model(batch_data['cross-fts']).reshape(*batch_data['shorty']['vals'].shape).detach().cpu().numpy()
                inds[ctr:ctr+bsz] = batch_data['shorty']['inds'].detach().cpu().numpy()
                
                ctr += bsz
                del batch_data
            torch.cuda.empty_cache() 
            return csr_matrix((data.ravel(), inds.ravel(), indptr), (datalen, numy))


# ### Evaluators

# In[1]:


import datetime


# In[ ]:


class PrecEvaluator():
    def __init__(self, model: GenericModel, dataloader, predictor, filter_mat = None, K=5, metric='P'):
        self.K = K
        self.metric = metric
        self.dataloader = dataloader
        self.predictor = predictor
        self.filter_mat = filter_mat
        self.model = model
        self.best_score = -9999999

    def __call__(self, epoch: int = -1, loss: float = -1.0, out_dir: str = None, name: str = ''):
        print(f'Evaluating {name} {["after epoch %d: "%epoch, ": "][name == ""]}')
        self.predictor.K = max(self.predictor.K, 100)
        score_mat = self.predictor(self.model, self.dataloader)
        if self.filter_mat is not None:
            _filter(score_mat, self.filter_mat, copy=False)
        res = printacc(score_mat, X_Y=self.dataloader.dataset.labels, K=max(5, self.K))
        recall = xc_metrics.recall(score_mat, tst_X_Y, k=100)*100
        print(f'Recall@100: {"%.2f"%recall[99]}')
        
        if out_dir is not None:
            out_file = f'{out_dir}/{[name+"_", ""][name == ""]}evaluation.tsv'
            print(f'dumping evaluation in {out_file}')
            if not os.path.exists(out_file):
                print('\t'.join(['epoch', 'time', 'loss', *[f'{metric}@1' for metric in res.index], *[f'{metric}@{self.K}' for metric in res.index], 'R@100']), file=open(out_file, 'w'))
            with open(out_file, 'a') as f:
                print('\t'.join([str(epoch), str(datetime.datetime.now()), str("%.4E"%loss), *["%.2f"%val for val in res['1'].values], *["%.2f"%val for val in res[str(self.K)].values], "%.2f"%recall[99]]), file=f)
        
        score = res[str(self.K)][self.metric]
        if score > self.best_score:
            print(f'found best model with score : {"%.4f"%score}')
            self.best_score = score
            if out_dir is not None:
                print(f'saving best model in {out_dir}')
                self.model.save(out_dir)
        return score


# ### Experimental blocks

# In[ ]:


from transformers import RobertaModel, AutoConfig
import functools
import operator
from multiprocessing import Pool

def b_kmeans_dense(labels_features, index, metric='cosine', tol=1e-4, leakage=None):
    with torch.no_grad():
        n = labels_features.shape[0]
        if labels_features.shape[0] == 1:
            return [index]
        cluster = np.random.randint(low=0, high=labels_features.shape[0], size=(2))

        while cluster[0] == cluster[1]:
            cluster = np.random.randint(
                low=0, high=labels_features.shape[0], size=(2))
        _centeroids = labels_features[cluster]

        _similarity = torch.mm(labels_features, _centeroids.T)
        old_sim, new_sim = -1000000, -2

        while new_sim - old_sim >= tol:
            clustered_lbs = torch.split(torch.argsort(_similarity[:, 1]-_similarity[:, 0]), (_similarity.shape[0]+1)//2)
            _centeroids = F.normalize(torch.vstack([torch.mean(labels_features[x, :], axis=0) for x in clustered_lbs]))
            _similarity = torch.mm(labels_features, _centeroids.T)
            
            old_sim, new_sim = new_sim, sum([torch.sum(_similarity[indx, i]) for i, indx in enumerate(clustered_lbs)]).item()/n
        del _similarity
        return list(map(lambda x: index[x], clustered_lbs))

def cluster_labels(labels, clusters, num_nodes, splitter):
    start = time.time()
    min_splits = min(32, num_nodes)
    while len(clusters) < num_nodes:
        temp_cluster_list = functools.reduce(
            operator.iconcat,
            map(lambda x: splitter(labels[x], x),
                clusters), [])
        end = time.time()
        print(f"Total clusters {len(temp_cluster_list)}\tAvg. Cluster size {'%.2f'%(np.mean(list(map(len, temp_cluster_list))))}\tTotal time {'%.2f'%(end-start)} sec")
        clusters = temp_cluster_list
        del temp_cluster_list
    with Pool(5) as p:
        cpu_labels = labels.to('cpu')
        while len(clusters) < num_nodes:
            temp_cluster_list = functools.reduce(
                operator.iconcat,
                p.starmap(
                    splitter,
                    map(lambda cluster: (cpu_labels[cluster], cluster),
                        clusters)
                ), [])
            end = time.time()
            print("Total clusters {}".format(len(temp_cluster_list)),
                  "Avg. Cluster size {}".format(
                      np.mean(list(map(len, temp_cluster_list)))),
                  "Total time {} sec".format(end-start))
            clusters = temp_cluster_list
            del temp_cluster_list
    return clusters

def cluster_dense_embs(embs, device='cpu', tree_depth = 18):
    print(f'device: {device}')
    clusters = cluster_labels(torch.tensor(embs).to(device), [torch.arange(embs.shape[0])], 2**(tree_depth-1), b_kmeans_dense)
    clustering_mat = sp.csr_matrix(  (np.ones(sum([len(c) for c in clusters])), 
                                     np.concatenate(clusters),
                                     np.cumsum([0, *[len(c) for c in clusters]])),
                                 shape=(len(clusters), embs.shape[0]))
    return clustering_mat


# In[ ]:


class DenoisedSiameseTripletOHNM(nn.Module):
    def __init__(self, model: GenericModel, numy, reduction='mean', margin=0.3, k=2, denoise_th=0.9, num_shortlist=50):
        super(DenoisedSiameseTripletOHNM, self).__init__()
        self.model = model
        self.numy = numy
        self.reduction = reduction
        self.margin = margin
        self.k = k
        self.denoise_th = denoise_th
        self.num_shortlist = num_shortlist

    def forward(self, batch_data):
        xembs = self.model.encode(batch_data['xfts'])
        yembs = self.model.encode(batch_data['yfts'])
        target = batch_data['targets']
        sim = xembs @ yembs.T
        sim_p = torch.gather(sim, 1, batch_data['pos-inds']).reshape(batch_data['batch_size'], -1)
        neg_sim = torch.where(target < 1e-6, sim, torch.full_like(sim, -100))
        neg_sim = torch.where(neg_sim < self.denoise_th, neg_sim, torch.full_like(sim, -1))
        
        _, indices = torch.topk(neg_sim, largest=True, dim=1, k=self.num_shortlist) # shortlist top num_shortlist negatives
        indices = indices.gather(1, torch.multinomial(torch.ones_like(indices).double(), self.k)) # uniformly sample k negatives from shortlist
        sim_n = sim.gather(1, indices) # model scores for final negatives
        
        loss = torch.max(torch.zeros_like(sim_p), sim_n - sim_p + self.margin)
        mask = torch.where(loss != 0, torch.ones_like(loss), torch.zeros_like(loss))
        prob = torch.softmax(sim_n * mask, dim=1)
        reduced_loss =  _reduce(loss * prob, self.reduction)
        
        del batch_data, xembs, yembs, sim, neg_sim, sim_p, sim_n, mask, prob, loss
        return reduced_loss


# In[ ]:


from typing import Iterator, Optional, Sequence, List, TypeVar, Generic, Sized
from torch.utils.data import Sampler
import gc

class MySampler(Sampler[int]):
    data_source: Sized

    def __init__(self, data_source, order):
        self.data_source = data_source
        self.order = order
        assert len(order) == len(data_source)

    def __iter__(self):
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.data_source)


# In[ ]:


class ClusterUpdater(nn.Module):
    def __init__(self, net, init_epoch=10, epoch_interval=5, cluster_size=32):
        super(ClusterUpdater, self).__init__()
        self.cmat = None
        self.net = net
        self.init_epoch = init_epoch
        self.epoch_interval = epoch_interval
        self.cluster_size = cluster_size

    def forward(self, epoch, dataloaders):
        if epoch>=self.init_epoch:
            if epoch%self.epoch_interval==0 or epoch==self.init_epoch:
                print('Updating cluster mat...')
                embs = self.net.get_embs(dataloaders[0].dataset, 'point')
                tree_depth = int(np.ceil(np.log(embs.shape[0]/self.cluster_size)/np.log(2)))+1
                self.cmat = cluster_dense_embs(embs, self.net._target_device, tree_depth).tocsr()
                del embs
                gc.collect()
            
            print('Updating dataloader...')
            cmat = self.cmat[np.random.permutation(self.cmat.shape[0])]
            batch_sampler = torch.utils.data.sampler.BatchSampler(MySampler(dataloaders[0].dataset, cmat.indices),
                                                                  params.batch_size, 
                                                                  False)
            dataloaders[0] = torch.utils.data.DataLoader(
                                        dataloaders[0].dataset,
                                        num_workers=dataloaders[0].num_workers,
                                        collate_fn=dataloaders[0].collate_fn,
                                        shuffle=False,
                                        batch_sampler=batch_sampler,
                                        pin_memory=dataloaders[0].pin_memory)

class ShuffleShortyUpdater(nn.Module):
    def __init__(self, shorty, init_epoch=10, epoch_interval=5, topk=1, num_shortlist=20):
        super(ShuffleShortyUpdater, self).__init__()
        self.shorty = shorty
        self.init_epoch = init_epoch
        self.epoch_interval = epoch_interval
        self.topk = topk
        self.num_shortlist = num_shortlist

    def forward(self, epoch, dataloaders):
        if epoch>=self.init_epoch and epoch%self.epoch_interval==0:
            print('Updating dataloader...')
            shorty = self.shorty.copy()
            shorty = _filter(shorty, dataloaders[0].dataset.labels)
            shorty = xclib.utils.sparse.retain_topk(shorty, k=self.num_shortlist)
            shorty.data = np.random.rand(shorty.nnz)
            num = shorty.shape[0]*self.topk
            rand_mat = csr_matrix((np.full(num, 1e-6), 
                                   np.random.choice(np.arange(shorty.shape[1]), size=num, replace=True), 
                                   range(0, num+1, self.topk)), (shorty.shape[0], shorty.shape[1]))
            shorty = xclib.utils.sparse.retain_topk(shorty+rand_mat, k=self.topk)
            dataloaders[0].dataset.shorty = shorty

class HardNegUpdater(nn.Module):
    def __init__(self, net, init_epoch=10, epoch_interval=5, topk=1, num_shortlist=20):
        super(HardNegUpdater, self).__init__()
        self.net = net
        self.shorty = None
        self.init_epoch = init_epoch
        self.epoch_interval = epoch_interval
        self.topk = topk
        self.num_shortlist = num_shortlist

    def forward(self, epoch, dataloaders):
        if epoch>=self.init_epoch:
            print('Updating dataloader...')
            if epoch%self.epoch_interval==0 or epoch==self.init_epoch:
                print('Updating shortlist...')
                xembs = self.net.get_embs(dataloaders[0].dataset, 'point')
                yembs = self.net.get_embs(dataloaders[0].dataset, 'label')
                es = exact_search({'data': yembs, 'query': xembs, 'K': max(100, self.num_shortlist), 'device': self.net._target_device})
                self.shorty = es.getnns_gpu()
                
                del xembs, yembs
                gc.collect()
                
            shorty = _filter(self.shorty.copy(), dataloaders[0].dataset.labels)
            shorty = xclib.utils.sparse.retain_topk(shorty, k=self.num_shortlist)
            shorty.data = np.random.rand(shorty.nnz)
            num = shorty.shape[0]*self.topk
            rand_mat = csr_matrix((np.full(num, 1e-6), 
                                   np.random.choice(np.arange(shorty.shape[1]), size=num, replace=True), 
                                   range(0, num+1, self.topk)), (shorty.shape[0], shorty.shape[1]))
            shorty = xclib.utils.sparse.retain_topk(shorty+rand_mat, k=self.topk)
            dataloaders[0].dataset.shorty = shorty
            
''' ------------------------------- Legacy Code -------------------------------'''
class BertDataset(torch.utils.data.Dataset):

    def __init__(self, point_texts, label_texts, labels, shorty=None, tokenizer_type='roberta-base', maxsize=512, data_root_dir=None):
        self.point_texts = point_texts
        self.label_texts = label_texts
        self.labels      = labels
        self.shorty      = shorty
        self.merge_fts_func = bert_fts_batch_to_tensor

        assert len(point_texts) == labels.shape[0], f'length of point texts ({len(point_texts)}) should be same as num rows of label correlation matrix ({labels.shape[0]})'
        assert len(label_texts) == labels.shape[1], f'length of label texts ({len(label_texts)}) should be same as num cols of label correlation matrix ({labels.shape[1]})'
        
        print("------ Some stats about the dataset ------")
        print(f'Num points : {len(point_texts)}')
        print(f'Num labels : {len(label_texts)}')
        print("------------------------------------------", end='\n\n')

        self.num_Y = self.labels.shape[1]
        self.num_X = self.labels.shape[0]
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type, do_lower_case=True)
        self.data_dir = f'{data_root_dir}/{tokenizer_type}_{maxsize}'
        try:
            print(f'trying to load pre-tokenized files from {self.data_dir} ...')
            self.point_encoded_dict = {'input_ids': np.load(f'{self.data_dir}/point_input_ids_{self.num_X}.npy'),
                                       'attention_mask': np.load(f'{self.data_dir}/point_attention_mask_{self.num_X}.npy')}
            self.label_encoded_dict = {'input_ids': np.load(f'{self.data_dir}/label_input_ids_{self.num_Y}.npy'),
                                       'attention_mask': np.load(f'{self.data_dir}/label_attention_mask_{self.num_Y}.npy')}
            print(f'successfully loaded pre-tokenized files from {self.data_dir}')
        except:
            print(f'unable to load pre-tokenized files from {self.data_dir}')
            print(f'creating tokenized files from raw texts...')
            
            start=time.time(); self.point_encoded_dict = self.convert(point_texts, maxsize); end=time.time()
            print(f'tokeinized points in {"%.2f"%(end-start)} s')
            start=time.time(); self.label_encoded_dict = self.convert(label_texts, maxsize); end=time.time()
            print(f'tokeinized labels in {"%.2f"%(end-start)} s')
            
            if not data_root_dir is None:
                print(f'saving tokenized files in {self.data_dir}')
                os.makedirs(self.data_dir, exist_ok=True)
                np.save(f'{self.data_dir}/point_input_ids_{self.num_X}.npy', self.point_encoded_dict['input_ids'])
                np.save(f'{self.data_dir}/point_attention_mask_{self.num_X}.npy', self.point_encoded_dict['attention_mask'])
                np.save(f'{self.data_dir}/label_input_ids_{self.num_Y}.npy', self.label_encoded_dict['input_ids'])
                np.save(f'{self.data_dir}/label_attention_mask_{self.num_Y}.npy', self.label_encoded_dict['attention_mask'])
        
    def convert(self, corpus, maxsize=512, bsz=100000):
        max_len = max(min(max([len(sen) for sen in corpus]), maxsize), 16)
        encoded_dict = {'input_ids': [], 'attention_mask': []}
        
        for ctr in tqdm(range(0, len(corpus), bsz)):
            temp = self.tokenizer.batch_encode_plus(
                    corpus[ctr:min(ctr+bsz, len(corpus))],  # Sentence to encode.
                    add_special_tokens = True,              # Add '[CLS]' and '[SEP]'
                    max_length = max_len,                   # Pad & truncate all sentences.
                    padding = 'max_length',
                    return_attention_mask = True,           # Construct attn. masks.
                    return_tensors = 'np',                  # Return numpy tensors.
                    truncation=True
            )
            encoded_dict['input_ids'].append(temp['input_ids'])
            encoded_dict['attention_mask'].append(temp['attention_mask'])
            
        encoded_dict['input_ids'] = np.vstack(encoded_dict['input_ids'])
        encoded_dict['attention_mask'] = np.vstack(encoded_dict['attention_mask'])
        return encoded_dict
    
    def transpose(self):
        print('Transposing dataset')
        self.labels = self.labels.T.tocsr()
        self.point_texts, self.label_texts = self.label_texts, self.point_texts
        self.point_encoded_dict, self.label_encoded_dict = self.label_encoded_dict, self.point_encoded_dict
        self.num_Y = self.labels.shape[1]
        self.num_X = self.labels.shape[0]

    def __getitem__(self, index):
        ret = {'index': index}
        ret['xfts'] = self.get_fts(index, 'point')
        temp = get_index_values(self.labels, index)
        ret['y'] = {'inds': temp[0], 'vals': temp[1]}
        
        if not self.shorty is None:
            temp = get_index_values(self.shorty, index)
            ret['shorty'] = {'inds': temp[0], 'vals': temp[1]}
        return ret
    
    def get_fts(self, index, source='label'):
        if source == 'label':
            encoded_dict = self.label_encoded_dict
        elif source == 'point':
            encoded_dict = self.point_encoded_dict
            
        if isinstance(index, int) or isinstance(index, np.int32):
            return {'input_ids': encoded_dict['input_ids'][index], 
                    'attention_mask': encoded_dict['attention_mask'][index]}
        else:
            return bert_fts_batch_to_tensor(encoded_dict['input_ids'][index],
                                            encoded_dict['attention_mask'][index])

    @property
    def num_instances(self):
        return self.labels.shape[0]

    @property
    def num_labels(self):
        return self.labels.shape[1]
    
    def __len__(self):
        return self.labels.shape[0]
