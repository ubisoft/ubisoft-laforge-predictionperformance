
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


class CodeForcesDataset(Dataset):
    def __init__(self, asts_path, problems_path, n_min_samples, N=None):
        super(CodeForcesDataset, self).__init__()

        asts = pd.read_pickle(asts_path)
        asts['id'] = asts['id'].astype(int)
        problems = pd.read_csv(problems_path)
        
        source_ids = asts['id']
        
        if N is not None:
            source_ids = source_ids.iloc[:N]
        
        self.source_ids = source_ids

        problems = problems[
            problems['id'].isin(source_ids)
            ]

        submissions_per_problem = problems.groupby('problem')['id'].count()
        problem_ids = submissions_per_problem[submissions_per_problem >= n_min_samples].index.array
        problems = problems[
            problems['problem'].isin(problem_ids)
        ]
        self.n_labels = len(problem_ids)
        self.label_map = {p: i for i, p in enumerate(problem_ids)}
        print(f'n_min_samples={n_min_samples}, n_labels={self.n_labels}, n_data={len(problems)}')

        self.problems = problems
        self.problems = self.problems.reset_index()
        self.problems['index'] = self.problems.index
        
        times_ids = set(problems['id'])
        self.blocks = {row['id']: row['blocks'] for i, row in asts.iterrows() if row['id'] in times_ids}
        print(f'{len(self.blocks)=}')
        del asts

        self.data = problems[['id', 'problem']].copy()
        self.data = list(self.data.itertuples(index=False, name=None))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        code_id, problem = self.data[i]
        # print(code_id)
        block = self.blocks[code_id]

        return block, self.label_map[problem]

# %%
def collate_to_lists(data):
    x = [x for x, c in data]
    target = torch.tensor([c for x, c in data]).reshape(-1)
    return x, target

def get_dataloaders_random(dataset, collate_fn, d=10, batchsize=32, pin_memory=True):
    n_eval = len(dataset) // d
    n_train = len(dataset) - 2*n_eval
    train_data, eval_data, test_data = random_split(dataset, [n_train, n_eval, n_eval])
    train_dl = DataLoader(train_data, batch_size=batchsize, shuffle=True, pin_memory=pin_memory, collate_fn=collate_fn)
    eval_dl = DataLoader(eval_data, batch_size=batchsize, shuffle=False, pin_memory=pin_memory, collate_fn=collate_fn)
    test_dl = DataLoader(test_data, batch_size=batchsize, shuffle=False, pin_memory=pin_memory, collate_fn=collate_fn)
    return train_dl, eval_dl, test_dl


def get_train_eval_test(asts_path, problems_path, dryrun, n_min_samples, N):
    print('Load data ...')
    data = CodeForcesDataset(
        asts_path, problems_path, n_min_samples,
        N = N if N is not None else 2000 if dryrun else None
    )

    seed_everything()

    batchsize = 32

    print('random split')
    train_dl, eval_dl, test_dl = get_dataloaders_random(data, collate_to_lists, batchsize=batchsize, pin_memory=True)
    
    print(f'{len(train_dl)=} {len(eval_dl)=} {len(test_dl)=}')
    print(f'{len(train_dl.dataset)=} {len(eval_dl.dataset)=} {len(test_dl.dataset)=}')

    return train_dl, eval_dl, test_dl

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input training specs.')
    parser.add_argument('--n_epoch', type=int, default=15)
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('--c', action='store_true')
    # parser.add_argument('--penalty', type=float, default=1.0)
    parser.add_argument('--l2', type=float, default=0.0)
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
    foldername = f'AstNN_{s}'
    if args.c:
        foldername = 'c/' + foldername
    
    if not args.dryrun:
        epoch_time = int(time.time())
        foldername += f'_{epoch_time}'

    foldername = 'classification/' + foldername
    Path(f'results/{foldername}').mkdir(exist_ok=args.dryrun, parents=True)

    #%%
    if args.c:
        asts_path = 'data/cf_c_asts.pkl'
        problems_path = 'data/cf_c_times.csv'
    else:
        asts_path = 'data/cf_cpp_asts.pkl'
        problems_path = 'data/cf_cpp_times.csv'
    n_min_samples = args.n_min_samples

    config = vars(args)
    config['script'] = 'astnn_train_class'
    config['problems_path'] = problems_path
    config['asts_path'] = asts_path
    

    train_dl, eval_dl, test_dl = get_train_eval_test(asts_path, problems_path, args.dryrun, n_min_samples, args.N)

    #assert False

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

    n_labels = train_dl.dataset.dataset.n_labels
    model = BatchProgramClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=100,
        vocab_size=max_tokens+1,
        encode_dim=128,
        label_size = n_labels,
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
        return current['accuracy'] < eval_stats['accuracy']

    train_stats, eval_stats, target_stats = train_model(
        model, optimizer, n_epoch, train_dl, eval_dl, None, device,
        domain_adaption=False, agg_train_stats=agg_class_train_stats,
        foldername=foldername, config=config,
        is_improv=is_improv)
    
    #%%
    print('Loading best model...')
    r = model.load_state_dict(torch.load(f'results/{foldername}/model.pt'))
    print(r)
    #%%
    _, y = test(model, train_dl, device, 'Train', agg_train_stats=agg_class_train_stats)
    plot_classification(y, 'train', foldername, target_names=None)
    #%%
    test_stats, y = test(model, test_dl, device, 'Test', agg_train_stats=agg_class_train_stats)
    plot_classification(y, 'test', foldername, target_names=None)

    with open(f'results/{foldername}/test_stats.txt', 'w') as f:
        f.write(str(test_stats))

    # %%
    plt.figure(figsize=(8,6))
    plt.plot(train_stats['accuracy'], label='Train Accuracy', color='tab:blue', linestyle='dashdot')
    plt.plot(eval_stats['accuracy'], label='Validation Accuracy', color='tab:orange', linestyle='dashdot')
    plt.xlabel(f'Epoch')
    plt.ylim((0,1))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(foldername)
    plt.tight_layout()
    plt.savefig(f'results/{foldername}/trainplot.png')
# %%
