#%%
from astnn_train_class import *
import argparse
import pandas as pd
from ast import literal_eval

#%%
if __name__ == '__main__':
    #%%
    parser = argparse.ArgumentParser(description='Input training specs.')
    parser.add_argument('--folder', type=str)

    #%%
    args = parser.parse_args()
    print(args)
    foldername = args.folder

    #%%
    foldername = 'results/classification/AstNN_10_1668138727'

    config = None
    with open(f'{foldername}/stats.txt', 'r') as f:
        config = literal_eval(f.readline())
    print(config)


    train_dl, eval_dl, test_dl = get_train_eval_test(config['asts_path'], config['problems_path'], config['dryrun'], config['n_min_samples'], config['N'])

    #%%
    #device = torch.device('cuda')
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

    n_labels = train_dl.dataset.dataset.n_labels
    model = BatchProgramClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=100,
        vocab_size=max_tokens+1,
        encode_dim=128,
        label_size = n_labels,
        batch_size=None, # set for each batch for different sized batches
        use_gpu=use_gpu,
        pretrained_weight=embeddings
    )
    model = model.to(device)
    
    #%%
    print('Loading model...')
    r = model.load_state_dict(torch.load(f'{foldername}/model.pt'))
    print(r)
    
    #%%
    test_stats, y = test(model, test_dl, device, 'Test', agg_train_stats=agg_class_train_stats)

    #%%
    #%%
    def get_target(dataset):
        if type(dataset) is Subset:
            subset = dataset # is Subset
            dataset = subset.dataset # full data set
            target = dataset.problems['time'].iloc[subset.indices]
        else:
            target = dataset.problems['time']
        return target
        
    zero_R_pred = get_target(train_dl.dataset).mean()
    Y = get_target(train_dl.dataset)
    train_zero_R = mean_squared_error(Y, np.full(len(Y), zero_R_pred), squared=False)
    Y = get_target(test_dl.dataset)
    test_zero_R = mean_squared_error(Y, np.full(len(Y), zero_R_pred), squared=False)
    print(f'{train_zero_R=:.2f}, {test_zero_R=:.2f}')

    #%%
    data = train_dl.dataset.dataset
    # just learning which problem, and predict mean
    train_times = data.problems.iloc[train_dl.dataset.indices]
    problem_meanruntime = train_times.groupby('problem')['time'].mean().reset_index().rename(columns={'time': 'meantime'})
    X = train_times.merge(problem_meanruntime)
    train_by_problem_zero_R = metrics.mean_squared_error(X['time'], X['meantime'], squared=False)
    print(f'{train_by_problem_zero_R=}')

    #%%
    # perfect case
    test_times = data.problems.iloc[test_dl.dataset.indices]
    X = test_times.merge(problem_meanruntime)
    test_by_problem_zero_R = metrics.mean_squared_error(X['time'], X['meantime'], squared=False)
    print(f'{test_by_problem_zero_R=}')

    #%%
    train_times = data.problems.iloc[train_dl.dataset.indices].copy()
    train_times['label'] = train_times['problem'].apply(lambda p: data.label_map[p])
    problem_meanruntime = train_times.groupby('label')['time'].mean().reset_index().rename(columns={'time': 'meantime'})
    X = train_times.merge(problem_meanruntime)
    train_by_problem_zero_R = metrics.mean_squared_error(X['time'], X['meantime'], squared=False)
    print(f'{train_by_problem_zero_R=}')

    #%%
    # learned case
    test_times = data.problems.iloc[test_dl.dataset.indices].copy()
    test_times['label'] = test_times['problem'].apply(lambda p: data.label_map[p])
    assert (test_times['label'] == y['y_true'].numpy()).all()
    test_times['label'] = y['y_pred'].numpy() # change to predicted instead of true labels
    X = test_times.merge(problem_meanruntime)
    test_by_problem_zero_R = metrics.mean_squared_error(X['time'], X['meantime'], squared=False)
    print(f'{test_by_problem_zero_R=}')

    #%%
    plt.figure(figsize=(4,4))
    plt.scatter(X['time'], X['meantime'], label=f'RMSE={test_by_problem_zero_R:.4f}', alpha=0.05)
    plt.xlabel('True Execution Time')
    plt.ylabel('Predicted Execution Time')
    plt.axline([0,0],[1,1], color='red')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{foldername}/predplot.png')
    plt.show()

    # [test_dl.dataset.dataset.label_map[test_dl.dataset.dataset.data[test_dl.dataset.indices[i]][1]] for i in range(3)]
    # == [test_dl.dataset[i][1] for i in range(3)]
    # == [test_dl.dataset.dataset[test_dl.dataset.indices[i]][1] for i in range(3)]
    #%%    
    train_stats = pd.read_csv(f'{foldername}/train_stats.csv')
    eval_stats = pd.read_csv(f'{foldername}/eval_stats.csv')

    plt.figure(figsize=(4,4))
    plt.plot(train_stats['accuracy'], label='Train Accuracy', color='tab:blue', linestyle='dashdot')
    plt.plot(eval_stats['accuracy'], label='Validation Accuracy', color='tab:orange', linestyle='dashdot')
    plt.xlabel(f'Epoch')
    plt.ylim((0,1))
    plt.legend(loc='center right')
    plt.tight_layout()
    plt.savefig(f'{foldername}/trainplot_paper.png')
# %%
