#%%
from astnn_train_reg import *
import argparse
import pandas as pd
import numpy as np
from ast import literal_eval
from astnn_train import UbiDataset

def block2seq_rec(b, seq):
    for el in b:
        if isinstance(el, list):
            block2seq_rec(el, seq)
        else:
            seq.append(el)

def block2seq(b):
    seq = []
    block2seq_rec(b, seq)
    return seq

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
    #foldername = 'results/regression/p_AstNN_20_1668260805'
    #foldername = 'results/regression/r_AstNN_20_1668177279'

    config = None
    with open(f'{foldername}/stats.txt', 'r') as f:
        config = literal_eval(f.readline())
    print(config) 

    train_dl, eval_dl, test_dl = get_train_eval_test(config['asts_path'], config['times_path'], config['dryrun'], config['N'], config['split_method'], config['n_min_samples'])

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
    
    #%%
    print('Loading model...')
    r = model.load_state_dict(torch.load(f'{foldername}/model.pt'))
    print(r)
    
    #%%
    train_encodings = []
    with torch.no_grad():
        for x, l in tqdm(train_dl):
            model.batch_size = len(l) # label length
            e = model.encode(x)
            train_encodings.append(e)

    train_encodings = torch.cat(train_encodings)
    train_encodings = train_encodings / train_encodings.norm(dim=1)[:, None]
    #%%
    test_similarity = []
    test_loss = []
    with torch.no_grad():
        for x, y_true in tqdm(test_dl):
            model.batch_size = len(y_true) # label length
            e = model.encode(x)
            e_normed = e / e.norm(dim=1)[:, None]
            
            y_pred = model.hidden2val(e)

            s = np.abs(train_encodings @ e_normed.T)
            s = s.sort(descending=True, dim=0).values[:100, :].T
            #s = s.max(dim=0).values
            test_similarity.append(s)

            test_loss.append(torch.abs(y_true - y_pred))

    test_similarity = torch.cat(test_similarity)
    test_similarity = test_similarity.numpy()
    test_loss = torch.cat(test_loss).numpy()
    np.save(f'{foldername}/test_similarity.npy', test_similarity)
    np.save(f'{foldername}/test_loss.npy', test_loss)

    #%%

    #%%
    train_seqs = []
    for xs, l in tqdm(train_dl):
        for x in xs:
            train_seqs.append(block2seq(x))
            
    #%%
    from difflib import SequenceMatcher
    test_similarity = []
    test_loss = []
    with torch.no_grad():
        for xs, y_true in tqdm(test_dl):
            for x in xs:
                test_seq = block2seq(x)
                s = []
                for train_seq in (train_seqs[::20]):
                    sm = SequenceMatcher(None, train_seq, test_seq)
                    r = sm.ratio()
                    s.append(r)
                test_similarity.append(sorted(s, reverse=True)[:100])

            model.batch_size = len(y_true) # label length
            y_pred = model(xs)

            test_loss.append(torch.abs(y_true - y_pred))

    test_similarity = np.array(test_similarity) 
    test_loss = torch.cat(test_loss).numpy()

    np.save(f'{foldername}/test_input_similarity.npy', test_similarity)
    np.save(f'{foldername}/test_input_loss.npy', test_loss)

    #%%

    #%%
    test_seqs = []
    for xs, l in tqdm(test_dl):
        for x in xs:
            test_seqs.append(block2seq(x))
    #%%
    from difflib import SequenceMatcher
    ubidata = UbiDataset('data/ubi_asts.pkl')
    def target_collate_to_lists(data):
        left = [l for l, r, c in data]
        right = [r for l, r, c in data]
        label = torch.tensor([c for l, r, c in data])
        return left, right, label
    target_dl = DataLoader(ubidata, batch_size=8, pin_memory=True, shuffle=True, collate_fn=target_collate_to_lists)

    target_similarity = []
    with torch.no_grad():
        for xs, _, _ in tqdm(target_dl):
            for x in xs:
                s1 = block2seq(x)
                s = []
                for s2 in (test_seqs):
                    sm = SequenceMatcher(None, s1, s2)
                    r = sm.ratio()
                    s.append(r)
                target_similarity.append(sorted(s, reverse=True)[:100])

    target_similarity = np.array(target_similarity)
    
    np.save(f'{foldername}/target_input_similarity.npy', target_similarity)

    #%%
    p_foldername = 'results/regression/p_AstNN_20_1668260805'
    r_foldername = 'results/regression/r_AstNN_20_1668177279'

    #%%
    foldername = r_foldername

    #%%
    # enconding similarity
    #test_similarity = np.load(f'{foldername}/test_similarity.npy')
    #test_loss = np.load(f'{foldername}/test_loss.npy')
    #plt.scatter(test_similarity.max(axis=1), test_loss, alpha=0.1)
    #plt.xlabel('Similarity to Train')
    #plt.ylabel('Loss')
    #plt.savefig(f'{foldername}/similarity_vs_loss.png')

    #%%
    plt.figure(figsize=(3,4))
    test_similarity = np.load(f'{foldername}/test_input_similarity.npy')
    test_loss = np.load(f'{foldername}/test_input_loss.npy')
    plt.scatter(test_similarity.max(axis=1), test_loss, alpha=0.05)
    plt.xlabel('Input Similarity to Train\nWithin problem')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(f'{foldername}/input_similarity_vs_loss.png')
    
    #%%
    target_similarity = np.load(f'{foldername}/target_input_similarity.npy')
    plt.boxplot([test_similarity.max(axis=1), target_similarity.max(axis=1)])
    plt.xticks(ticks=[1,2], labels=['test', 'target'])
    plt.title('Input Similarity')
    plt.savefig(f'{foldername}/input_similarity.png')

    #%%
    plt.figure(figsize=(2,4))
    p_test_similarity = np.load(f'{p_foldername}/test_input_similarity.npy')
    r_test_similarity = np.load(f'{r_foldername}/test_input_similarity.npy')
    plt.boxplot([p_test_similarity.max(axis=1), r_test_similarity.max(axis=1)])
    plt.xticks(ticks=[1,2], labels=['New pr.', 'Within pr.'], rotation=25)
    #plt.title('Input Similarity')
    plt.ylabel('Input similarity')
    plt.tight_layout()
    plt.savefig(f'{r_foldername}/similarity_r_vs_p.pdf')


    #%%
    np.quantile(target_similarity, 0.75) - np.quantile(target_similarity, 0.25)
    #%%
    
    fig, axs = plt.subplots(ncols=2, figsize=(6,4), sharey=True)
    p_test_similarity = np.load(f'{p_foldername}/test_input_similarity.npy')
    p_test_loss = np.load(f'{p_foldername}/test_input_loss.npy')
    r_test_similarity = np.load(f'{r_foldername}/test_input_similarity.npy')
    r_test_loss = np.load(f'{r_foldername}/test_input_loss.npy')

    axs[0].scatter(p_test_similarity.max(axis=1), p_test_loss, alpha=0.05)
    axs[0].set_ylabel('Error')
    axs[0].set_xlabel('Input Similarity\nOn new problems')
    axs[1].scatter(r_test_similarity.max(axis=1), r_test_loss, alpha=0.05)
    axs[1].set_xlabel('Input Similarity\nWithin problems')
    for a in axs:
        a.set_xticks([0.25, 0.5, 0.75, 1.0])
    plt.subplots_adjust(wspace=0, hspace=0)
    
    plt.tight_layout()
    plt.savefig(f'{r_foldername}/similarity_r_vs_p_scatter.png')
# %%
