#%%
from astnn_train_reg import *
import argparse
import pandas as pd
from ast import literal_eval

#%%
#name = 'p_AstNN_20_1668260805'
name = 'r_AstNN_20_1668177279'
foldername = f'results/regression/{name}'

config = None
with open(f'{foldername}/stats.txt', 'r') as f:
    config = literal_eval(f.readline())
print(config)

# %%
train_dl, eval_dl, test_dl = get_train_eval_test(
    config['asts_path'], config['times_path'], config['dryrun'],
    config['N'], config['split_method'], config['n_min_samples'])

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

model = BatchProgramRegressor(
    embedding_dim=embedding_dim,
    hidden_dim=100,
    vocab_size=max_tokens+1,
    encode_dim=128,
    batch_size=None, # set for each batch for different sized batches
    use_gpu=use_gpu,
    pretrained_weight=embeddings
)
model = model.to(device)
# %%
print('Loading best model...')
r = model.load_state_dict(torch.load(f'{foldername}/model.pt'))
print(r)
# %%
test_stats, y = test(model, test_dl, device, 'Test', agg_train_stats=agg_reg_train_stats, loss_func=nn.MSELoss())

#%%
train_stats = pd.read_csv(f'{foldername}/train_stats.csv')
eval_stats = pd.read_csv(f'{foldername}/eval_stats.csv')
train_zero_R = config['train_zero_R']
eval_zero_R = config['eval_zero_R']

#%%
plt.figure(figsize=(4,4))
plt.plot(train_stats['rmse'], label='Train', color='tab:blue', linestyle='dashdot')
plt.axhline(y=train_zero_R, label='Train ZeroR', color='tab:blue', linestyle='dotted')
plt.plot(eval_stats['rmse'], label='Validation', color='tab:orange', linestyle='dashdot')
plt.axhline(y=eval_zero_R, label='Val. ZeroR', color='tab:orange', linestyle='dotted')
plt.xlabel(f'Epoch')
plt.xticks(np.arange(0,21,2))
plt.legend(loc='center right')#, bbox_to_anchor=(1, 0.5))
plt.ylabel('Execution time RMSE')
plt.tight_layout()
plt.savefig(f'{foldername}/{name}_trainplot_paper.pdf')

#%%
rmse = test_stats['rmse']
plt.figure(figsize=(4,4))
plt.scatter(y['y_true'], y['y_pred'], label=f'RMSE={rmse:.4f}', alpha=0.05)
plt.xlabel('True Execution Time')
plt.ylabel('Predicted Execution Time')
plt.axline([0,0],[1,1], color='red')
#plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(f'{foldername}/{name}_predplot.png')
plt.show()

# %%
zero_R_pred = get_target(train_dl.dataset).mean()
print(f'{zero_R_pred=}')
y = get_target(train_dl.dataset)
train_zero_R = mean_squared_error(y, np.full(len(y), zero_R_pred), squared=False)
y = get_target(eval_dl.dataset)
eval_zero_R = mean_squared_error(y, np.full(len(y), zero_R_pred), squared=False)
y = get_target(test_dl.dataset)
test_zero_R = mean_squared_error(y, np.full(len(y), zero_R_pred), squared=False)
print(f'{train_zero_R=:.2f}, {eval_zero_R=:.2f}, {test_zero_R=:.2f}')
# %%
