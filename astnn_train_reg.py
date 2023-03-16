
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
from sklearn.metrics import mean_squared_error

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

def get_target(dataset):
    if type(dataset) is Subset:
        subset = dataset # is Subset
        dataset = subset.dataset # full data set
        target = dataset.times['time'].iloc[subset.indices]
    else:
        target = dataset.times['time']
    return target

class CodeForcesDataset(Dataset):
    def __init__(self, asts_path, time_path, N=None, n_min_samples=None):
        super(CodeForcesDataset, self).__init__()

        asts = pd.read_pickle(asts_path)
        print(f'{len(asts)=}')
        asts['id'] = asts['id'].astype(int)
        times = pd.read_csv(time_path)
        times['time'] = times['time'].astype(float)
        
        source_ids = asts['id']
        
        self.source_ids = source_ids

        self.times = times[
            times['id'].isin(source_ids)
            ]

        if n_min_samples is not None:
            submissions_per_problem = self.times.groupby('problem')['id'].count()
            problem_ids = submissions_per_problem[submissions_per_problem >= n_min_samples].index.array
            self.times = self.times[
                self.times['problem'].isin(problem_ids)
            ]
        print(f'{len(self.times)=}')
            
        if N is not None:
            self.source_ids = self.source_ids.iloc[:N]
            self.times = self.times[
                self.times['id'].isin(self.source_ids)
                ]
            print(f'After N: {len(self.times)}=')

        self.times = self.times.reset_index()
        self.times['index'] = self.times.index
        
        times_ids = set(self.times['id'])
        self.blocks = {row['id']: row['blocks'] for i, row in asts.iterrows() if row['id'] in times_ids}
        print(f'{len(self.blocks)=}')
        del asts

        self.data = self.times[['id', 'time']].copy()
        self.data = list(self.data.itertuples(index=False, name=None))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        code_id, target =  self.data[i]
        # print(code_id)
        block = self.blocks[code_id]

        return block, target

# %%
def collate_to_lists(data):
    x = [x for x, c in data]
    target = torch.tensor([c for x, c in data]).reshape(-1,1)
    return x, target

def get_dataloaders_random(dataset, collate_fn, d=10, batchsize=32, pin_memory=True):
    n_eval = len(dataset) // d
    n_train = len(dataset) - 2*n_eval
    train_data, eval_data, test_data = random_split(dataset, [n_train, n_eval, n_eval])
    
    times = train_data.dataset.times
    train_times = times.iloc[train_data.indices]
    eval_times = times.iloc[eval_data.indices]
    test_times = times.iloc[test_data.indices]
    print('train_times', train_times['time'].mean(), len(train_times['problem'].unique()))
    print('eval_times', eval_times['time'].mean(), len(eval_times['problem'].unique()))
    print('test_times', test_times['time'].mean(), len(test_times['problem'].unique()))

    train_dl = DataLoader(train_data, batch_size=batchsize, shuffle=True, pin_memory=pin_memory, collate_fn=collate_fn)
    eval_dl = DataLoader(eval_data, batch_size=batchsize, shuffle=False, pin_memory=pin_memory, collate_fn=collate_fn)
    test_dl = DataLoader(test_data, batch_size=batchsize, shuffle=False, pin_memory=pin_memory, collate_fn=collate_fn)
    return train_dl, eval_dl, test_dl

def get_dataloaders_author_problem(dataset, collate_fn, grouping_cols, batchsize=32, pin_memory=True):
    grouping = dataset.times[grouping_cols].drop_duplicates()
    #print(grouping, len(grouping))
    problems = dataset.times['problem'].unique()
    #print(problems, len(problems))

    N = len(grouping)
    n_eval = N // 10
    n_train = N - 2*n_eval

    train = grouping.sample(n_train)
    nottrain = grouping.drop(train.index)
    eval = nottrain.sample(frac=0.5)
    test = nottrain.drop(eval.index)
    assert len(train) + len(eval) + len(test) == len(grouping)

    def get_times_subset(times, grouping):
        return times.merge(grouping, how='inner') # inner join

    train_times = get_times_subset(dataset.times, train)
    eval_times = get_times_subset(dataset.times, eval)
    test_times = get_times_subset(dataset.times, test)
    assert len(train_times) + len(eval_times) + len(test_times) == len(dataset.times)
    print('train_times', train_times['time'].mean(), len(train_times['problem'].unique()))
    print('eval_times', eval_times['time'].mean(), len(eval_times['problem'].unique()))
    print('test_times', test_times['time'].mean(), len(test_times['problem'].unique()))

    #print('train_times', dataset.times.loc[dataset.times['problem'].isin(train['problem']), 'time'].mean())
    #print('eval_times', dataset.times.loc[dataset.times['problem'].isin(eval['problem']), 'time'].mean())
    #print('test_times', dataset.times.loc[dataset.times['problem'].isin(test['problem']), 'time'].mean())

    # just to make sure, check for disjointness
    train_ids = set(train_times['id'])
    eval_ids = set(eval_times['id'])
    test_ids = set(test_times['id'])
    assert len(train_ids.intersection(eval_ids)) == 0 and len(train_ids.intersection(test_ids)) == 0 and len(eval_ids.intersection(test_ids)) == 0

    train_set = Subset(dataset, list(train_times['index']))

    train_dl = DataLoader(
        train_set,
        batch_size=batchsize, shuffle=True, pin_memory=pin_memory,
        collate_fn=collate_fn)

    eval_dl = DataLoader(
        Subset(dataset, list(eval_times['index'])),
        batch_size=batchsize, shuffle=False, pin_memory=pin_memory,
        collate_fn=collate_fn)

    test_dl = DataLoader(
        Subset(dataset, list(test_times['index'])),
        batch_size=batchsize, shuffle=False, pin_memory=pin_memory,
        collate_fn=collate_fn)

    train_ixs = set(dataset.times.iloc[train_dl.dataset.indices].index)
    eval_ixs = set(dataset.times.iloc[eval_dl.dataset.indices].index)
    test_ixs = set(dataset.times.iloc[test_dl.dataset.indices].index)
    assert train_ixs.difference(eval_ixs) and train_ixs.difference(test_ixs) and eval_ixs.difference(test_ixs)

    return train_dl, eval_dl, test_dl

def get_train_eval_test(asts_path, times_path, dryrun, N, split_method, n_min_samples):
    print('Load data ...')
    data = CodeForcesDataset(
        asts_path, times_path,
        N = N if N is not None else 2000 if dryrun else None,
        n_min_samples=n_min_samples
    )
    #print(f'{get_target(data).mean()=}')

    seed_everything()

    batchsize = 32

    if split_method == 'r':
        print('random split')
        train_dl, eval_dl, test_dl = get_dataloaders_random(data, collate_to_lists, batchsize=batchsize, pin_memory=True)
    else:
        grouping_cols = {'p': ['problem'], 'ap':['author', 'problem']}[split_method]
        print(f'{grouping_cols=}')
        train_dl, eval_dl, test_dl = get_dataloaders_author_problem(data, collate_fn=collate_to_lists, batchsize=batchsize, pin_memory=True, grouping_cols=grouping_cols)
        
    print(f'{len(train_dl)=} {len(eval_dl)=} {len(test_dl)=}')
    print(f'{len(train_dl.dataset)=} {len(eval_dl.dataset)=} {len(test_dl.dataset)=}')

    return train_dl, eval_dl, test_dl

#%%
if __name__ == '__main__':
    #%%
    parser = argparse.ArgumentParser(description='Input training specs.')
    parser.add_argument('--n_epoch', type=int, default=15)
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--c', action='store_true')
    parser.add_argument('--split_method', type=str, default='p') # p, ap, r
    # parser.add_argument('--penalty', type=float, default=1.0)
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--N', type=int, default=None)
    parser.add_argument('--n_min_samples', type=int, default=None)

    #%%
    args = parser.parse_args()
    print(args)

    #%%
    #input = [
    #    '--n_min_samples', '128',
    #    '--split_method', 'p'
    #    ]
    #args = parser.parse_args(input)
    #print(args)

    #%%
    n_epoch = args.n_epoch if not args.dryrun else 1

    #%%
    s = 'dryrun' if args.dryrun else str(n_epoch)
    foldername = f'{args.split_method}_AstNN_{s}'
    if args.c:
        foldername = 'c/' + foldername
    
    if not args.dryrun:
        epoch_time = int(time.time())
        foldername += f'_{epoch_time}'

    foldername = 'regression/' + foldername
    Path(f'results/{foldername}').mkdir(exist_ok=args.dryrun, parents=True)

    #%%
    if args.c:
        asts_path = 'data/cf_c_asts.pkl'
        times_path = 'data/cf_c_times.csv'
    else:
        asts_path = 'data/cf_cpp_asts.pkl'
        times_path = 'data/cf_cpp_times.csv'

    config = vars(args)
    config['script'] = 'astnn_train_reg'
    config['times_path'] = times_path
    config['asts_path'] = asts_path
    

    train_dl, eval_dl, test_dl = get_train_eval_test(asts_path, times_path, args.dryrun, args.N, args.split_method, args.n_min_samples)

    #%%
    zero_R_pred = get_target(train_dl.dataset).mean()
    print(f'{zero_R_pred=}')
    y = get_target(train_dl.dataset)
    train_zero_R = mean_squared_error(y, np.full(len(y), zero_R_pred), squared=False)
    y = get_target(eval_dl.dataset)
    eval_zero_R = mean_squared_error(y, np.full(len(y), zero_R_pred), squared=False)
    print(f'{train_zero_R=:.2f}, {eval_zero_R=:.2f}')
    config['train_zero_R'] = train_zero_R
    config['eval_zero_R'] = eval_zero_R

    #%%
    zero_R_pred = get_target(eval_dl.dataset).mean()
    print(f'{zero_R_pred=}')
    y = get_target(eval_dl.dataset)
    eval_zero_R_2 = mean_squared_error(y, np.full(len(y), zero_R_pred), squared=False)
    print(f'{eval_zero_R_2=:.2f}')
    
    #%%
    # just learning which problem, and predict mean
    train_times = train_dl.dataset.dataset.times.iloc[train_dl.dataset.indices]
    problem_meanruntime = train_times.groupby('problem')['time'].mean().reset_index().rename(columns={'time': 'meantime'})
    X = train_times.merge(problem_meanruntime)
    train_by_problem_zero_R = metrics.mean_squared_error(X['time'], X['meantime'], squared=False)
    print(f'{train_by_problem_zero_R=}')
    config['train_by_problem_zero_R'] = train_by_problem_zero_R
    if args.split_method == 'r':
        eval_times = eval_dl.dataset.dataset.times.iloc[eval_dl.dataset.indices]
        X = eval_times.merge(problem_meanruntime)
        eval_by_problem_zero_R = metrics.mean_squared_error(X['time'], X['meantime'], squared=False)
        print(f'{eval_by_problem_zero_R=}')
        config['eval_by_problem_zero_R'] = eval_by_problem_zero_R

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

    model = BatchProgramRegressor(
        embedding_dim=embedding_dim,
        hidden_dim=100,
        vocab_size=max_tokens+1,
        encode_dim=128,
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
        return current['rmse'] > eval_stats['rmse']

    train_stats, eval_stats, target_stats = train_model(
        model, optimizer, n_epoch, train_dl, eval_dl, None, device,
        domain_adaption=False, agg_train_stats=agg_reg_train_stats,
        foldername=foldername, config=config, loss_func=nn.MSELoss(),
        is_improv=is_improv)
    
    #%%
    print('Loading best model...')
    r = model.load_state_dict(torch.load(f'results/{foldername}/model.pt'))
    print(r)
    #%%
    _, y = test(model, train_dl, device, 'Train', agg_train_stats=agg_reg_train_stats, loss_func=nn.MSELoss())
    plot_regression(y, 'train', foldername)
    #%%
    test_stats, y = test(model, test_dl, device, 'Test', agg_train_stats=agg_reg_train_stats, loss_func=nn.MSELoss())
    plot_regression(y, 'test', foldername)
    with open(f'results/{foldername}/test_stats.txt', 'w') as f:
        f.write(str(test_stats))

    # %%
    plt.figure(figsize=(8,6))
    plt.plot(train_stats['rmse'], label='Train RMSE', color='tab:blue', linestyle='dashdot')
    plt.axhline(y=train_zero_R, label='Train ZeroR', color='tab:blue', linestyle='dotted')
    plt.plot(eval_stats['rmse'], label='Validation RMSE', color='tab:orange', linestyle='dashdot')
    plt.axhline(y=eval_zero_R, label='Eval ZeroR', color='tab:orange', linestyle='dotted')
    plt.xlabel(f'Epoch')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(foldername)
    plt.tight_layout()
    plt.savefig(f'results/{foldername}/trainplot.png')
# %%
