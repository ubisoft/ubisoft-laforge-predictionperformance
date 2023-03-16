from torch.utils.data import Dataset, DataLoader, Subset, random_split
import pandas as pd
import numpy as np

def get_class_dist(labels):
    class_dist = labels.value_counts() / len(labels)
    return class_dist

def get_majority_class_prop(labels):
    class_dist = get_class_dist(labels)
    print(labels.value_counts())
    print(class_dist)
    majority_class_prop = class_dist.max()
    return majority_class_prop

def undersample_subset(subset, labels):
    subset.indices = np.array(subset.indices)
    labels = labels[subset.indices]
    class_counts = labels.value_counts()
    min_count = class_counts.min()
    new_indices = []
    for l, c in class_counts.iteritems():
        class_indices = pd.Series(subset.indices[labels == l])
        if c > min_count:
            new_class_indicices = np.array(class_indices.sample(min_count, replace=False))
        else:
            new_class_indicices = np.array(class_indices)
        new_indices.append(new_class_indicices)

    subset.indices = np.hstack(new_indices) # make sure to shuffle for training

def oversample_subset(subset, labels):
    subset.indices = np.array(subset.indices)
    labels = labels[subset.indices]
    class_counts = labels.value_counts()
    maj_count = class_counts.max()
    new_indices = []
    for l, c in class_counts.iteritems():
        class_indices = pd.Series(subset.indices[labels == l])
        if c < maj_count:
            # repeat until we have to sample
            repeated_class_indices = np.array(class_indices.repeat(maj_count // c))
            rem = maj_count - len(repeated_class_indices)
            sampled_class_indices = np.array(class_indices.sample(rem, replace=False))
            new_class_indicices = np.hstack((repeated_class_indices, sampled_class_indices))
        else:
            new_class_indicices = np.array(class_indices)
        new_indices.append(new_class_indicices)
    
    subset.indices = np.hstack(new_indices) # make sure to shuffle for training


def get_pair_subset(pairs, grouping):
    return pairs.merge(grouping, how='inner') # inner join

def get_dataloaders_author_problem(
    dataset, labels, grouping_cols, collate_fn, batchsize=32, pin_memory=False, sampling='no'):

    # 80% train 10% eval 10% test
    # based on solutions of author to problem, because we only compare those

    grouping = dataset.pairs[grouping_cols].drop_duplicates()

    N = len(grouping)
    n_eval = N // 10 
    n_train = N - 2*n_eval

    train = grouping.sample(n_train)
    nottrain = grouping.drop(train.index)
    eval = nottrain.sample(frac=0.5)
    test = nottrain.drop(eval.index)
    assert len(train) + len(eval) + len(test) == len(grouping)

    train_pairs = get_pair_subset(dataset.pairs, train)
    eval_pairs = get_pair_subset(dataset.pairs, eval)
    test_pairs = get_pair_subset(dataset.pairs, test)
    assert len(train_pairs) + len(eval_pairs) + len(test_pairs) == len(dataset.pairs)
    
    # just to make sure, check for disjointness
    train_ids = set(list(train_pairs['before_source_ix']) + list(train_pairs['after_source_ix']))
    eval_ids = set(list(eval_pairs['before_source_ix']) + list(eval_pairs['after_source_ix']))
    test_ids = set(list(test_pairs['before_source_ix']) + list(test_pairs['after_source_ix']))
    assert len(train_ids.intersection(eval_ids)) == 0 and len(train_ids.intersection(test_ids)) == 0 and len(eval_ids.intersection(test_ids)) == 0

    train_ids = set(list(train_pairs['before_id']) + list(train_pairs['after_id']))
    eval_ids = set(list(eval_pairs['before_id']) + list(eval_pairs['after_id']))
    test_ids = set(list(test_pairs['before_id']) + list(test_pairs['after_id']))
    assert len(train_ids.intersection(eval_ids)) == 0 and len(train_ids.intersection(test_ids)) == 0 and len(eval_ids.intersection(test_ids)) == 0

    train_set = Subset(dataset, list(train_pairs['index']))
    if sampling == 'over':
        oversample_subset(train_set, labels)
    if sampling == 'under':
        undersample_subset(train_set, labels)

    train_dl = DataLoader(
        train_set,
        batch_size=batchsize, shuffle=True, pin_memory=pin_memory,
        collate_fn=collate_fn)

    eval_dl = DataLoader(
        Subset(dataset, list(eval_pairs['index'])),
        batch_size=batchsize, shuffle=False, pin_memory=pin_memory,
        collate_fn=collate_fn)

    test_dl = DataLoader(
        Subset(dataset, list(test_pairs['index'])),
        batch_size=batchsize, shuffle=False, pin_memory=pin_memory,
        collate_fn=collate_fn)

    train_ixs = set(dataset.pairs.iloc[train_dl.dataset.indices].index)
    eval_ixs = set(dataset.pairs.iloc[eval_dl.dataset.indices].index)
    test_ixs = set(dataset.pairs.iloc[test_dl.dataset.indices].index)
    assert train_ixs.difference(eval_ixs) and train_ixs.difference(test_ixs) and eval_ixs.difference(test_ixs)

    return train_dl, eval_dl, test_dl


# pair in test can be reverse pair in train
def get_dataloaders_random(dataset, labels, collate_fn, d=10, batchsize=32, pin_memory=False, sampling='no'):
    # 80% train 10% eval 10% test if d=10
    n_eval = len(dataset) // d
    n_train = len(dataset) - 2*n_eval
    train_data, eval_data, test_data = random_split(dataset, [n_train, n_eval, n_eval])
    if sampling == 'over':
        oversample_subset(train_data, labels)
    if sampling == 'under':
        undersample_subset(train_data, labels)
    train_dl = DataLoader(train_data, batch_size=batchsize, shuffle=True, pin_memory=pin_memory, collate_fn=collate_fn)
    eval_dl = DataLoader(eval_data, batch_size=batchsize, shuffle=False, pin_memory=pin_memory, collate_fn=collate_fn)
    test_dl = DataLoader(test_data, batch_size=batchsize, shuffle=False, pin_memory=pin_memory, collate_fn=collate_fn)
    return train_dl, eval_dl, test_dl

# pair in test can NOT be reverse pair in train
def get_dataloaders_random_pairs(dataset, labels, collate_fn, d=10, batchsize=32, pin_memory=False, sampling='no'):
    # 80% train 10% eval 10% test if d=10
    pairs = dataset.pairs
    pairs['id_pair'] = [f'{a},{b}' if a < b else f'{b},{a}' for b, a in zip(pairs['before_id'], pairs['after_id'])]
    id_pairs = pd.Series(pairs['id_pair'].unique())

    n_eval = len(id_pairs) // d 
    n_train = len(id_pairs) - 2*n_eval
    id_pairs = id_pairs.sample(frac=1)

    train_ids = id_pairs.iloc[:n_train]
    eval_ids = id_pairs.iloc[n_train:(n_train+n_eval)]
    test_ids = id_pairs.iloc[(n_train+n_eval):]

    train_pairs = pairs[pairs['id_pair'].isin(train_ids)]
    eval_pairs = pairs[pairs['id_pair'].isin(eval_ids)]
    test_pairs = pairs[pairs['id_pair'].isin(test_ids)]
    assert len(train_pairs) + len(eval_pairs) + len(test_pairs) == len(pairs)

    train_set = Subset(dataset, list(train_pairs['index']))
    if sampling == 'over':
        oversample_subset(train_set, labels)
    if sampling == 'under':
        undersample_subset(train_set, labels)

    train_dl = DataLoader(
            train_set,
            batch_size=batchsize, shuffle=True, pin_memory=pin_memory,
            collate_fn=collate_fn)

    eval_dl = DataLoader(
        Subset(dataset, list(eval_pairs['index'])),
        batch_size=batchsize, shuffle=False, pin_memory=pin_memory,
        collate_fn=collate_fn)

    test_dl = DataLoader(
        Subset(dataset, list(test_pairs['index'])),
        batch_size=batchsize, shuffle=False, pin_memory=pin_memory,
        collate_fn=collate_fn)

    
    train_ixs = set(dataset.pairs.iloc[train_dl.dataset.indices].index)
    eval_ixs = set(dataset.pairs.iloc[eval_dl.dataset.indices].index)
    test_ixs = set(dataset.pairs.iloc[test_dl.dataset.indices].index)
    assert train_ixs.difference(eval_ixs) and train_ixs.difference(test_ixs) and eval_ixs.difference(test_ixs)
    if sampling == 'no':
        assert len(train_ixs) + len(eval_ixs) + len(test_ixs) == len(pairs)

    return train_dl, eval_dl, test_dl

# %%
