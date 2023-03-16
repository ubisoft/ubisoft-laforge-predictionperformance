import imp
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from astnn_model import *
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from torch.nn.utils.rnn import PackedSequence

def get_next_target_batch(target_dl, target_dl_iter):
    try:
        return target_dl_iter, target_dl_iter.next()
    except StopIteration:
        # restart
        target_dl_iter = iter(target_dl)
        return target_dl_iter, target_dl_iter.next()

def train(model, optimizer, train_dl, device, epoch, n_epoch,
    loss_func=nn.CrossEntropyLoss(),
    domain_adaption=False, target_dl=None,
    agg_train_stats=None):

    model.train()
    train_stats = {'loss': [], 'out': [], 'y_true': []}

    if domain_adaption:
        assert target_dl is not None
        target_dl_iter = iter(target_dl)
        train_stats['domain_pred'] = []
        train_stats['domain_true'] = []

    cross_entropy = nn.CrossEntropyLoss()
    
    for i, batch in (pbar := tqdm(enumerate(train_dl), total=len(train_dl), desc=f'Epoch {epoch}')):
        optimizer.zero_grad()

        if domain_adaption:
            p = (epoch-1.) / n_epoch 
            alpha = 2. / (1. + np.exp(-10 * p)) - 1 # goes from 0 to 1, exponent controlls how slow
        else:
            alpha = None

        if isinstance(model, ASTNN):
            model.batch_size = len(batch[-1]) # label length

        if len(batch) == 3:
            # comparator
            left, right, label = batch
            if isinstance(left, torch.Tensor) or isinstance(left, PackedSequence):
                left = left.to(device)
            if isinstance(right, torch.Tensor) or isinstance(right, PackedSequence):
                right = right.to(device)
            label = label.to(device)

            out = model(left, right, alpha) if domain_adaption else model(left, right)
        else:
            # classifier, regressor
            x, label = batch
            if isinstance(x, torch.Tensor) or isinstance(x, PackedSequence):
                x = x.to(device)
            label = label.to(device)

            out = model(x, alpha) if domain_adaption else model(x)

        out_domain = None
        if isinstance(out, tuple) and len(out) == 2:
            # domain adaption possible
            out, out_domain = out

        loss = loss_func(out, label)

        train_stats['loss'].append(loss.item())
        train_stats['y_true'].append(label.detach().cpu())
        train_stats['out'].append(out.detach().cpu())

        if domain_adaption:
            source_domain_label = torch.zeros(len(label)).long().to(device) # label 0
            source_domain_loss = cross_entropy(out_domain, source_domain_label)
            domain_loss = source_domain_loss
            train_stats['domain_pred'].append(out_domain.detach().cpu())
            train_stats['domain_true'].append(source_domain_label.detach().cpu())

            # target domain
            target_dl_iter, target_batch = get_next_target_batch(target_dl, target_dl_iter)
            
            if isinstance(model, ASTNN):
                model.batch_size = len(target_batch[-1]) # label length

            if len(target_batch) == 3:
                # comparator
                left_target, right_target, label_target = target_batch
                if isinstance(left_target, torch.Tensor) or isinstance(left_target, PackedSequence):
                    left_target = left_target.to(device)
                if isinstance(right_target, torch.Tensor) or isinstance(right_target, PackedSequence):
                    right_target = right_target.to(device)

                _, out_domain_target = model(left_target, right_target, alpha)
            else:
                # classifier, regressor
                x_target, _ = target_batch
                if isinstance(x_target, torch.Tensor):
                    x_target = x_target.to(device)

                _, out_domain_target = model(x_target, alpha)

            target_domain_label = torch.ones(len(label_target)).long().to(device) # label 1

            train_stats['domain_pred'].append(out_domain_target.detach().cpu())
            train_stats['domain_true'].append(target_domain_label.detach().cpu())

            target_domain_loss = cross_entropy(out_domain_target, target_domain_label)
            domain_loss += target_domain_loss

            loss += domain_loss


        loss.backward()
        optimizer.step()

        if (i % (len(train_dl)//100 + 1) == 0):
            pbar.set_postfix({'loss': loss.item()})

    stats = agg_train_stats(train_stats)
    return stats

def test(model, eval_dl, device,
    name = '',
    loss_func=nn.CrossEntropyLoss(),
    agg_train_stats=None):

    model.eval()
    eval_stats = {'loss': [], 'out': [], 'y_true': []}
    
    with torch.no_grad():
        for i, batch in (pbar := tqdm(enumerate(eval_dl), total=len(eval_dl), desc=name)):
            if isinstance(model, ASTNN):
                model.batch_size = len(batch[-1]) # label length

            if len(batch) == 3:
                # comparator
                left, right, label = batch
                if isinstance(left, torch.Tensor) or isinstance(left, PackedSequence):
                    left = left.to(device)
                if isinstance(right, torch.Tensor) or isinstance(right, PackedSequence):
                    right = right.to(device)
                label = label.to(device)

                out = model(left, right)
            else:
                # classifier, regressor
                x, label = batch
                if isinstance(x, torch.Tensor) or isinstance(x, PackedSequence):
                    x = x.to(device)
                label = label.to(device)

                out = model(x)

            if isinstance(out, tuple) and len(out) == 2:
                # domain adaption possible
                out, _ = out

            loss = loss_func(out, label)

            eval_stats['loss'].append(loss.item())
            eval_stats['y_true'].append(label.detach().cpu())
            eval_stats['out'].append(out.detach().cpu())

    stats = agg_train_stats(eval_stats)
    return stats

def agg_class_comp_train_stats(train_stats):
    agg = {}
    y = {}

    y_score = torch.cat(train_stats['out']).softmax(dim=1)
    y_true = torch.cat(train_stats['y_true'])
    loss = torch.tensor(train_stats['loss']) # avg. loss for each batch
    avg_loss = loss.sum().item() / len(loss)
    
    y_pred = y_score.argmax(axis=1)

    assert y_score.shape[1] == 3
    y_diff_true = y_true != 1
    y_diff_score = torch.stack((y_score[:,1], y_score[:,0]+y_score[:,2]), dim=1)
    y_diff_pred = y_diff_score.argmax(axis=1)

    agg['avg_loss'] = avg_loss
    agg['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    agg['diff_accuracy'] = metrics.accuracy_score(y_diff_true, y_diff_pred)
    agg['diff_F1'] = metrics.f1_score(y_diff_true, y_diff_pred)
    agg['diff_auc'] = metrics.roc_auc_score(y_diff_true, y_diff_pred)
    agg['diff_ap'] = metrics.average_precision_score(y_diff_true, y_diff_pred)
    agg['diff_mcc'] = metrics.matthews_corrcoef(y_diff_true, y_diff_pred)

    y['y_true'] = y_true
    y['y_pred'] = y_pred
    y['y_score'] = y_score

    y['y_diff_true'] = y_diff_true
    y['y_diff_pred'] = y_diff_pred
    y['y_diff_score'] = y_diff_score

    acc = agg['accuracy']
    f1 = agg['diff_F1']
    auc = agg['diff_auc']
    mcc = agg['diff_mcc']
    print(f'loss: {avg_loss:.4f}, acc: {acc:.4f}, diff_f1: {f1:.4f}, diff_auc: {auc:.4f}, diff_mmc: {mcc:.4f}')

    return agg, y

def agg_class_binary_train_stats(train_stats):
    agg = {}
    y = {}

    y_score = torch.cat(train_stats['out']).softmax(dim=1)
    y_true = torch.cat(train_stats['y_true'])
    loss = torch.tensor(train_stats['loss']) # avg. loss for each batch
    avg_loss = loss.sum().item() / len(loss)
    
    y_pred = y_score.argmax(axis=1)

    assert y_score.shape[1] == 2

    agg['avg_loss'] = avg_loss
    agg['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    agg['F1'] = metrics.f1_score(y_true, y_pred)
    agg['auc'] = metrics.roc_auc_score(y_true, y_pred)
    agg['ap'] = metrics.average_precision_score(y_true, y_pred)
    agg['mcc'] = metrics.matthews_corrcoef(y_true, y_pred)

    y['y_true'] = y_true
    y['y_pred'] = y_pred
    y['y_score'] = y_score
    
    acc = agg['accuracy']
    f1 = agg['F1']
    auc = agg['auc']
    mcc = agg['mcc']
    print(f'loss: {avg_loss:.4f}, acc: {acc:.4f}, f1: {f1:.4f}, auc: {auc:.4f}, mcc: {mcc:.4f}')

    return agg, y


def agg_class_train_stats(train_stats):
    agg = {}
    y = {}

    y_score = torch.cat(train_stats['out']).softmax(dim=1)
    y_true = torch.cat(train_stats['y_true'])
    loss = torch.tensor(train_stats['loss']) # avg. loss for each batch
    avg_loss = loss.sum().item() / len(loss)
    
    y_pred = y_score.argmax(axis=1)

    assert y_score.shape[1] > 2

    agg['avg_loss'] = avg_loss
    agg['accuracy'] = metrics.accuracy_score(y_true, y_pred)

    y['y_true'] = y_true
    y['y_pred'] = y_pred
    
    acc = agg['accuracy']
    print(f'loss: {avg_loss:.4f}, acc: {acc:.4f}')

    return agg, y

def agg_reg_train_stats(train_stats):
    agg = {}
    y = {}

    y_pred = torch.cat(train_stats['out'])
    y_true = torch.cat(train_stats['y_true'])
    loss = torch.tensor(train_stats['loss']) # avg. MSE loss for each batch
    avg_loss = loss.sum().item() / len(loss)

    agg['avg_loss'] = avg_loss
    rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
    agg['rmse'] = rmse

    y['y_true'] = y_true
    y['y_pred'] = y_pred

    print(f'loss: {avg_loss:.4f}, rmse: {rmse:.4f}')

    return agg, y

def plot_classification(stats, name, foldername, target_names):
    classification_report = metrics.classification_report(stats['y_true'], stats['y_pred'], target_names=target_names)
    print(classification_report)
    reports = [classification_report]
    
    if target_names is not None and len(target_names) <= 10: 
        plt.figure()
        cm = metrics.confusion_matrix(stats['y_true'], stats['y_pred'])
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot()
        plt.title(foldername)
        plt.savefig(f'results/{foldername}/{name}_confusion.png')

    y_score = None
    y_pred = None
    y_true = None
    if 'y_score' in stats and stats['y_score'].shape[1] == 2:
        y_score = stats['y_score']
        y_pred = stats['y_pred']
        y_true = stats['y_true']
    elif 'y_diff_true' in stats:
        y_score = stats['y_diff_score']
        y_pred = stats['y_diff_pred']
        y_true = stats['y_diff_true']

    if y_score is not None:
        # get metrics for predicting performance difference, label 1 is same performance
        if 'y_diff_true' in stats:
            target_names = ['same perf.', 'sign.perf.diff.']
            diff_classification_report = metrics.classification_report(y_true, y_pred, target_names=target_names)
            reports.append(diff_classification_report)
            print(diff_classification_report)
    
        plt.figure()
        metrics.RocCurveDisplay.from_predictions(y_true, y_score[:,1])
        plt.title(foldername)
        plt.savefig(f'results/{foldername}/{name}_roc.png')
        
        plt.figure()
        metrics.PrecisionRecallDisplay.from_predictions(y_true, y_score[:,1])
        plt.title(foldername)
        plt.savefig(f'results/{foldername}/{name}_precrec.png')


    with open(f'results/{foldername}/{name}_report.txt', 'w') as f:
        f.write('\n\n'.join(reports))


def plot_regression(stats, name, foldername):
    rmse = metrics.mean_squared_error(stats['y_true'], stats['y_pred'], squared=False)
    plt.figure()
    plt.scatter(stats['y_true'], stats['y_pred'], label=f'RMSE={rmse:.4f}', alpha=0.1)
    plt.xlabel('True Execution Time')
    plt.ylabel('Predicted Execution Time')
    plt.axline([0,0],[1,1], color='red')
    plt.title(foldername)
    plt.savefig(f'results/{foldername}/{name}_predictionplot.png')


#%%
import time
import pandas as pd
def train_model(model, optimizer, n_epoch, train_dl, eval_dl, target_dl, device,
    domain_adaption=False, agg_train_stats=None,
    foldername=None, config=None, loss_func=nn.CrossEntropyLoss(),
    is_improv=None):

    t0 = time.time()
    print('Start training ...')

    train_stats = []
    eval_stats = []
    target_stats = []

    r, _ = test(model, train_dl, device, 'Train', agg_train_stats=agg_train_stats, loss_func=loss_func)
    train_stats.append(r)

    r, _ = test(model, eval_dl, device, 'Eval', agg_train_stats=agg_train_stats, loss_func=loss_func)
    eval_stats.append(r)

    if domain_adaption:
        r, _ = test(model, target_dl, device, 'Target', agg_train_stats=agg_train_stats, loss_func=loss_func)
        target_stats.append(r)

    try:
        current_best = None
        for epoch in range(n_epoch):
            epoch_stats, _ = train(
                model, optimizer, train_dl, device,
                epoch+1, n_epoch,
                loss_func=loss_func,
                domain_adaption=domain_adaption, target_dl=target_dl,
                agg_train_stats=agg_train_stats
                )
            train_stats.append(epoch_stats)

            r, _ = test(model, eval_dl, device, 'Eval', agg_train_stats=agg_train_stats, loss_func=loss_func)
            eval_stats.append(r)

            if is_improv is not None:
                if is_improv(current_best, eval_stats[-1]):
                    current_best = eval_stats[-1]
                    print('Best model so far. Save Model...')
                    torch.save(model.state_dict(), f'results/{foldername}/model.pt')

            if domain_adaption:
                r, _ = test(model, target_dl, device, 'Target', agg_train_stats=agg_train_stats, loss_func=loss_func)
                target_stats.append(r)
                
    except KeyboardInterrupt:
        print('Training interrupted.')
    
    if is_improv is None:
        print('Save Model...')
        torch.save(model.state_dict(), f'results/{foldername}/model.pt')

    walltime = time.time() - t0
    days = int(walltime // 86400)
    walltime = walltime % 86400
    duration_str = f'Training time: {days} days ' + time.strftime('%H:%M:%S', time.gmtime(walltime))
    print(duration_str)
    
    train_stats = pd.DataFrame(train_stats)
    train_stats.to_csv(f'results/{foldername}/train_stats.csv')
    
    eval_stats = pd.DataFrame(eval_stats)
    eval_stats.to_csv(f'results/{foldername}/eval_stats.csv')

    if domain_adaption:
        target_stats = pd.DataFrame(target_stats)
        target_stats.to_csv(f'results/{foldername}/target_stats.csv')

    
    with open(f'results/{foldername}/stats.txt', 'w') as f:
        f.write('\n\n'.join([str(config), duration_str]))

    return train_stats, eval_stats, target_stats