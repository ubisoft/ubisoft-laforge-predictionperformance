
#%%
import torch
from torch.optim import Adamax
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
from gensim.models.word2vec import Word2Vec
import argparse
from dataloader_splitting import *
from pathlib import Path
import matplotlib.pyplot as plt
import time

from astnn_model import *
from train_test import *

#%%
def seed_everything(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_labels(dataset):
    if isinstance(dataset, UbiDataset):
        labels = pd.Series([l.item() for _,_, l in dataset.data])
    elif type(dataset) is Subset:
        subset = dataset # is Subset
        dataset = subset.dataset # full data set
        labels = dataset.pairs['label'].iloc[subset.indices]
    else:
        labels = dataset.pairs['label']
    return labels

class CodeForcesDataset(Dataset):
    def __init__(self, asts_path, pairs_path, N=None, n_min_samples=None):
        super(CodeForcesDataset, self).__init__()

        asts = pd.read_pickle(asts_path)
        asts['id'] = asts['id'].astype(int)
        pairs = pd.read_csv(pairs_path)
        print(f'{len(asts)=}')

        self.source_ids = asts[['id', 'problem']]

        if n_min_samples is not None:
            submissions_per_problem = self.source_ids[['id', 'problem']].groupby('problem')['id'].count()
            problem_ids = submissions_per_problem[submissions_per_problem >= n_min_samples].index.array
            self.source_ids = self.source_ids[
                self.source_ids['problem'].isin(problem_ids)
            ]
        
        self.source_ids = self.source_ids.sort_values('problem')
        self.source_ids = self.source_ids['id'] # sorted by problem
        print(f'{len(self.source_ids)=}')

        if N is not None:
            self.source_ids = self.source_ids.iloc[:N]
            print(f'After N: {len(self.source_ids)=}')
        

        print(f'Total number of pairs: {len(pairs)}')
        pairs = pairs[
            pairs['before_id'].isin(self.source_ids) &
            pairs['after_id'].isin(self.source_ids)
            ]
        print(f'Used number of pairs: {len(pairs)}')

        self.pairs = pairs
        self.pairs = self.pairs.reset_index()
        self.pairs['index'] = self.pairs.index
        print('Class count:', self.pairs.groupby('label')['index'].count())
        
        pair_ids = set(pairs['before_id']).union(set(pairs['after_id']))
        self.blocks = {row['id']: row['blocks'] for i, row in asts.iterrows() if row['id'] in pair_ids}
        print(f'{len(self.blocks)=}')
        del asts

        self.data = pairs[['before_id', 'after_id', 'label']].copy()
        self.data['label'] = self.data['label'] + 1 
        self.data = list(self.data.itertuples(index=False, name=None))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        before_id, after_id, label =  self.data[i]
        
        before = self.blocks[before_id]
        after = self.blocks[after_id]

        return before, after, label



class UbiDataset(Dataset):
    def __init__(self, ubi_data_path, ix=None):
        super(UbiDataset, self).__init__()

        self.ubi_data = pd.read_pickle(ubi_data_path)
        if ix is not None:
            self.ubi_data = self.ubi_data.iloc[ix]

        self.data = []
        for _, row in self.ubi_data.iterrows():
            before = row['blocks_after']
            after = row['blocks_before']
            is_regression_fix = row['is_regression_fix']

            # label is -1,0,1 map to 0,1,2 for before faster, same, slower
            # label 1 means improvement, after is faster
            if not is_regression_fix:
                label = torch.tensor([0+1])
                self.data.append((before, after, label))
                self.data.append((after, before, label))
            else:
                label = torch.tensor([1+1])
                self.data.append((before, after, label))
                
                label = torch.tensor([-1+1])
                self.data.append((after, before, label))

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

# %%
def collate_to_lists(data):
    left = [l for l, r, c in data]
    right = [r for l, r, c in data]
    label = torch.tensor([c for l, r, c in data])
    return left, right, label


def get_train_eval_test_target(asts_path, pairs_path, ubi_path, dryrun, N, split_method, sampling, n_min_samples):
    print('Load data ...')
    data = CodeForcesDataset(
        asts_path, pairs_path,
        N = N if N is not None else 2000 if dryrun else None,
        n_min_samples = n_min_samples
    )

    seed_everything()

    if split_method == 'r':
        print('random split')
        train_dl, eval_dl, test_dl = get_dataloaders_random(data, get_labels(data), collate_to_lists, batchsize=32, pin_memory=True, sampling=sampling)
    elif split_method == 'rp':
        print('random pair split')
        train_dl, eval_dl, test_dl = get_dataloaders_random_pairs(data, get_labels(data), collate_to_lists, batchsize=32, pin_memory=True, sampling=sampling)
    else:
        grouping_cols = {'p': ['problem'], 'ap':['author', 'problem']}[split_method]
        print(f'{grouping_cols=}')
        train_dl, eval_dl, test_dl = get_dataloaders_author_problem(data, get_labels(data), collate_fn=collate_to_lists, batchsize=32, pin_memory=True, grouping_cols=grouping_cols, sampling=sampling)
        
    print(f'{len(train_dl)=} {len(eval_dl)=} {len(test_dl)=}')

    if ubi_path is not None:
        ubidata = UbiDataset(ubi_path)
        target_dl = DataLoader(ubidata, batch_size=8, pin_memory=True, shuffle=True, collate_fn=collate_to_lists)
        print(f'{len(target_dl)=}')
    else:
        target_dl = None

    return train_dl, eval_dl, test_dl, target_dl

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input training specs.')
    parser.add_argument('--n_epoch', type=int, default=15)
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--c', action='store_true')
    parser.add_argument('--split_method', type=str, default='p') # p, ap, r, rp
    parser.add_argument('--pairs_name', type=str)
    # parser.add_argument('--penalty', type=float, default=1.0)
    parser.add_argument('--sampling', type=str, default='no') # no, over, under
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--ubi', action='store_true')
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--n_min_samples', type=int, default=None)

    #%%
    args = parser.parse_args()
    print(args)

    #%%
    # input = [
    #     '--dryrun',
    #     '--split_method', 'ap'
    #     ]
    # args = parser.parse_args(input)
    # print(args)

    #%%
    n_epoch = args.n_epoch if not args.dryrun else 1

    #%%
    s = 'dryrun' if args.dryrun else str(n_epoch)
    foldername = f'{args.split_method}_AstNN_{s}'
    if args.c:
        foldername = 'c/' + foldername

    if args.ubi:
        foldername += '_ubi'
    
    if not args.dryrun:
        epoch_time = int(time.time())
        foldername += f'_{epoch_time}'

    Path(f'results/{foldername}').mkdir(exist_ok=args.dryrun, parents=True)

    #%%
    pairs_path = f'data/{args.pairs_name}.csv'
    if args.c:
        asts_path = 'data/cf_c_asts.pkl'
    else:
        asts_path = 'data/cf_cpp_asts.pkl'

    config = vars(args)
    config['script'] = 'astnn_train'
    config['pairs_path'] = pairs_path
    config['asts_path'] = asts_path
    

    if args.ubi:
        ubi_path = 'data/ubi_asts.pkl'
        config['ubi_path'] = ubi_path
    else:
        ubi_path = None

    train_dl, eval_dl, test_dl, target_dl = get_train_eval_test_target(asts_path, pairs_path, ubi_path, args.dryrun, args.N, args.split_method, args.sampling, args.n_min_samples)

    #assert False

    #%%
    print('Get majority class ...')
    train_zero_R = get_majority_class_prop(get_labels(train_dl.dataset))
    eval_zero_R = get_majority_class_prop(get_labels(eval_dl.dataset))
    if args.ubi:
        target_zero_R = get_majority_class_prop(get_labels(target_dl.dataset))
        print(f'{train_zero_R=:.2f}, {eval_zero_R=:.2f}, {target_zero_R=:.2f}')
    else:
        print(f'{train_zero_R=:.2f}, {eval_zero_R=:.2f}')

    #%%
    device = torch.device('cuda')
    seed_everything()

    #%%
    if args.c:
        word2vec = Word2Vec.load('data/cf_c_w2v').wv
    else:
        word2vec = Word2Vec.load('data/cf_cpp_w2v').wv

    max_tokens = word2vec.syn0.shape[0]
    embedding_dim = word2vec.syn0.shape[1]
    embeddings = np.zeros((max_tokens + 1, embedding_dim), dtype='float32')
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    model = BatchProgramComparator(
        embedding_dim=embedding_dim,
        hidden_dim=100,
        vocab_size=max_tokens+1,
        encode_dim=128,
        label_size=3,
        batch_size=None, # set for each batch for different sized batches
        use_gpu=True,
        pretrained_weight=embeddings
    )
    model = model.to(device)

    # %%
    optimizer = Adamax(model.parameters(), weight_decay=args.l2)

    def is_improv(current, eval_stats):
        if current is None:
            return True
        return current['diff_F1'] < eval_stats['diff_F1']

    train_stats, eval_stats, target_stats = train_model(
        model, optimizer, n_epoch, train_dl, eval_dl, target_dl, device,
        domain_adaption=args.ubi, agg_train_stats=agg_class_comp_train_stats,
        foldername=foldername, config=config,
        is_improv=is_improv)
    
    #%%
    print('Loading best model...')
    r = model.load_state_dict(torch.load(f'results/{foldername}/model.pt'))
    print(r)
    #%%
    target_names = ['slower', 'same', 'faster']
    test_stats, y = test(model, test_dl, device, 'Test', agg_train_stats=agg_class_comp_train_stats)
    plot_classification(y, 'test', foldername, target_names)
    
    with open(f'results/{foldername}/test_stats.txt', 'w') as f:
        f.write(str(test_stats))

    if args.ubi:
        target_stats, y = test(model, target_dl, device, 'target', agg_train_stats=agg_class_comp_train_stats)
        plot_classification(y, 'target', foldername, target_names)
        with open(f'results/{foldername}/target_stats.txt', 'w') as f:
            f.write(str(target_stats))

    # %%
    plt.figure(figsize=(8,6))
    plt.plot(train_stats['accuracy'], label='Train Accuracy', color='tab:blue', linestyle='dashdot')
    plt.plot(train_stats['diff_auc'], label='Train Diff. AUC', color='tab:blue')
    plt.axhline(y=train_zero_R, label='Train ZeroR', color='tab:blue', linestyle='dotted')
    plt.plot(eval_stats['accuracy'], label='Validation Accuracy', color='tab:orange', linestyle='dashdot')
    plt.plot(eval_stats['diff_auc'], label='Validation Diff. AUC', color='tab:orange')
    plt.axhline(y=eval_zero_R, label='Eval ZeroR', color='tab:orange', linestyle='dotted')
    if args.ubi:
        plt.plot(target_stats['accuracy'], color='tab:green', label='Target Accuracy', linestyle='dashdot')
        plt.plot(target_stats['diff_auc'], label='Target Diff. AUC', color='tab:green')
        plt.axhline(y=target_zero_R, label='Target ZeroR', color='tab:green', linestyle='dotted')
    plt.xlabel(f'Epoch')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0,1)
    plt.title(foldername)
    plt.tight_layout()
    plt.savefig(f'results/{foldername}/trainplot.png')
# %%
