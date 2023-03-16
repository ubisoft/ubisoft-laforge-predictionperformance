#%%
%load_ext autoreload
%autoreload 2
#%%
from astnn_train import *
import argparse
import pandas as pd
from ast import literal_eval

#%%
#name = 'p_AstNN_5_1669101384'
name = 'rp_AstNN_5_1668535298' # for transfer learning
#name = 'rp_AstNN_5_ubi_1668774038'
foldername = f'results/{name}'

config = None
with open(f'{foldername}/stats.txt', 'r') as f:
    config = literal_eval(f.readline())
print(config)

# %%

if config['ubi']:
    ubi_path = 'data/ubi_asts.pkl'
else:
    ubi_path = None

train_dl, eval_dl, test_dl, target_dl = get_train_eval_test_target(
    config['asts_path'], config['pairs_path'], ubi_path,
    config['dryrun'], config['N'], config['split_method'], config['sampling'], config['n_min_samples'])


# --- Load Model

# %%
device = torch.device('cpu')
seed_everything()
use_gpu = device.type=='cuda'
use_gpu

#%%
if config['c']:
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
    use_gpu=use_gpu,
    pretrained_weight=embeddings
)
model = model.to(device)

print('Loading best model...')
r = model.load_state_dict(torch.load(f'{foldername}/model.pt'))
print(r)

# --- Rerun test

# %%
test_stats, y = test(model, test_dl, device, 'Test', agg_train_stats=agg_class_comp_train_stats)

#%%
test_stats, y = test(model, target_dl, device, 'Test', agg_train_stats=agg_class_comp_train_stats)

# --- Transfer Learning

#%%
ubi_path = 'data/ubi_asts.pkl'
ubi_data = UbiDataset(ubi_path)

get_majority_class_prop(get_labels(ubi_data))

#%%
n_ubi = len(ubi_data.ubi_data)
ixs = np.arange(n_ubi)
np.random.seed(0)
np.random.shuffle(ixs)

n_train = n_ubi // 10 * 8
train_ixs = ixs[:n_train]
test_ixs = ixs[n_train:]

ubi_train = UbiDataset(ubi_path, ix=train_ixs)
ubi_test = UbiDataset(ubi_path, ix=test_ixs)

ubi_train_dl = DataLoader(ubi_train, batch_size=32, pin_memory=True, shuffle=True, collate_fn=collate_to_lists)
ubi_test_dl = DataLoader(ubi_test, batch_size=32, pin_memory=True, shuffle=True, collate_fn=collate_to_lists)
        
#%%

# %%
seed_everything()
ubi_foldername = name + '/ubi_transfer'
Path(f'results/{ubi_foldername}').mkdir(exist_ok=True, parents=True)
#%%

optimizer = Adamax(model.parameters())

def is_improv(current, eval_stats):
    if current is None:
        return True
    return current['diff_F1'] < eval_stats['diff_F1']


n_epoch=5
train_stats, test_stats, _ = train_model(
    model, optimizer, n_epoch, ubi_train_dl, ubi_test_dl, None, device,
    domain_adaption=False, agg_train_stats=agg_class_comp_train_stats,
    foldername=ubi_foldername, config=config,
    is_improv=is_improv)
# %%
train_stats
#%%
test_stats
#%%
r = model.load_state_dict(torch.load(f'results/{ubi_foldername}/model.pt'))
print(r)

#%%
_, y = test(model, ubi_test_dl, device, 'Test', agg_train_stats=agg_class_comp_train_stats)

#%%

plt.figure(figsize=(4,4))
plt.plot(train_stats['diff_auc'], label='Target Train AUC', color='tab:blue')
plt.plot(train_stats['diff_F1'], label='Target Train F1', color='tab:blue', linestyle='dashdot')

plt.plot(test_stats['diff_auc'], label='Target Test AUC', color='tab:green')
plt.plot(test_stats['diff_F1'], label='Target Test F1', color='tab:green', linestyle='dashdot')
plt.xlabel(f'Epoch')
plt.legend(loc='lower right')
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(f'results/{ubi_foldername}/trainplot_target_paper.pdf')




# ---- Train plots

#%%
def get_zero_R(dl):
    y_true = (get_labels(dl.dataset) + 1).array
    y_score = np.full((len(y_true), 3), [0.,1.,0.])
    d={'y_true': (torch.tensor(y_true),), 'out': (torch.tensor(y_score),), 'loss': np.zeros(len(y_true))}
    agg, _ = agg_class_comp_train_stats(d)
    return agg

train_zero_R = get_zero_R(train_dl)
eval_zero_R = get_zero_R(eval_dl)
test_zero_R = get_zero_R(test_dl)

#%%
target_zero_R = get_zero_R(target_dl)

# %%
train_stats = pd.read_csv(f'{foldername}/train_stats.csv')
eval_stats = pd.read_csv(f'{foldername}/eval_stats.csv')

plt.figure(figsize=(4,4))
#plt.plot(train_stats['accuracy'], label='Train Accuracy', color='tab:blue', linestyle='dashdot')
plt.plot(train_stats['diff_auc'], label='Train AUC', color='tab:blue')
plt.plot(train_stats['diff_F1'], label='Train F1', color='tab:blue', linestyle='dashdot')

#plt.axhline(y=train_zero_R, label='Train ZeroR', color='tab:blue', linestyle='dotted')
#plt.plot(eval_stats['accuracy'], label='Validation Accuracy', color='tab:orange', linestyle='dashdot')
plt.plot(eval_stats['diff_auc'], label='Validation AUC', color='tab:orange')
plt.plot(eval_stats['diff_F1'], label='Validation F1', color='tab:orange', linestyle='dashdot')
#plt.axhline(y=eval_zero_R, label='Eval ZeroR', color='tab:orange', linestyle='dotted')
if config['ubi']:
    target_stats = pd.read_csv(f'{foldername}/target_stats.csv')
    #plt.plot(target_stats['accuracy'], color='tab:green', label='Target Accuracy', linestyle='dashdot')
    plt.plot(target_stats['diff_auc'], label='Target AUC', color='tab:green')
    plt.plot(target_stats['diff_F1'], label='Target F1', color='tab:green', linestyle='dashdot')
    #plt.axhline(y=target_zero_R, label='Target ZeroR', color='tab:green', linestyle='dotted')
plt.xlabel(f'Epoch')
plt.legend(loc='lower right')
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(f'{foldername}/{name}_trainplot_paper.pdf')
# %%
