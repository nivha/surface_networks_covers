# Based on code from https://github.com/bfortuner/pytorch_tiramisu

import os
import csv
import shutil
import time
import datetime
import torch
import torch.nn as nn
from torch.autograd import Variable


class Logger(object):
    def __init__(self, logfilepath, epoch_logfile, start_time, net_name):
        self.logfilepath = logfilepath
        self.epoch_logfilepath = epoch_logfile
        self.start_time = start_time
        self.net_name = net_name

        # init log (with columns names..)
        columns = ['date', 'time [sec]', 'split', 'epoch', 'batch', 'n_batches', 'loss', 'error', 'accuracy']
        self.add2csv(self.logfilepath, columns)
        columns = ['date', 'time [sec]', 'epoch', 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy']
        self.add2csv(self.epoch_logfilepath, columns)

    def add2csv(self, csvpath, fields):
        with open(csvpath, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def log(self, split, epoch, batch, n_batches, loss, error, verbose=True):
        acc = 1 - error
        fields = [
            datetime.datetime.now(), round(time.time() - self.start_time, 3),
            split, epoch, batch, n_batches, loss, error, acc,
        ]
        if batch % 50 == 0:
            self.add2csv(self.logfilepath, fields)

        if verbose:
            print(datetime.datetime.now(), 'gpu (%s) %s: epoch: %3d batch %3d / %3d: loss %6.4f acc %6.4f' % (
                os.environ['CUDA_VISIBLE_DEVICES'], self.net_name, epoch, batch, n_batches, loss, acc))

    def log_epoch(self, epoch, trn_loss, trn_error, val_loss, val_error):
        fields = [
            datetime.datetime.now(), round(time.time() - self.start_time, 3),
            epoch, trn_loss, val_loss, 1 - trn_error, 1 - val_error,
        ]
        self.add2csv(self.epoch_logfilepath, fields)

        if self.net_name == 'debug': return


def save_weights(dirpath, model, epoch, loss, err, val_err):
    weights_fname = 'weights-%d-%.3f-%.3f-%.3f.pth' % (epoch, loss, err, val_err)
    weights_fpath = os.path.join(dirpath, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss': loss,
            'error': err,
            'val_error': val_err,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, os.path.join(dirpath, 'latest.th'))


def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath)
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['error']))
    return startEpoch


def get_predictions(output_batch, max_index=True):
    tensor = output_batch.data
    bs, c, h, w = output_batch.size()
    if max_index:
        values, indices = tensor.max(1)
        indices = indices.view(bs, h, w)
        return indices
    else:
        return tensor


def error(preds, targets):
    assert preds.size() == targets.size()
    bs, h, w = preds.size()
    n_pixels = bs*h*w
    incorrect = preds.ne(targets).sum()
    err = incorrect.type(torch.get_default_dtype()) / n_pixels
    return err.item()


def train(model, trn_loader, optimizer, criterion, epoch, logger=None):
    model.train()
    trn_loss = 0
    trn_error = 0
    n_batches = len(trn_loader)
    for idx, data in enumerate(trn_loader):

        inputs = data[0].cuda()
        targets = data[1].cuda()

        optimizer.zero_grad()
        output = model(inputs.cuda())

        # handle case of tnet loss (output is a tuple in that case)
        if isinstance(output, tuple):
            output, trans = output
            n_batch = output.shape[0]
            mat_diff = torch.bmm(trans, trans.permute(0, 2, 1)) - torch.eye(3, device=trans.device)
            trans_loss = mat_diff.pow(2).sum() / n_batch
            loss = criterion(output, targets) + 1e-3 * trans_loss
        else:
            loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        trn_loss += loss.data.item()
        pred = get_predictions(output)
        trn_error += error(pred, targets.data)

        batch_avg_loss = trn_loss / (1 + idx)
        batch_avg_err = trn_error / (1 + idx)
        if logger is not None:
            logger.log('train', epoch, idx+1, n_batches, batch_avg_loss, batch_avg_err, verbose=True)

    trn_loss /= n_batches
    trn_error /= n_batches
    return trn_loss, trn_error


@torch.no_grad()
def test(model, test_loader, criterion, epoch=1, logger=None):
    if model._get_name() == 'FCN32s':
        model.eval()

    if hasattr(model, 'stn'):
        model.stn.eval()

    test_loss = 0
    test_error = 0
    n_batches = len(test_loader)
    for idx, data in enumerate(test_loader):
        inputs = data[0].cuda()
        targets = data[1].cuda()
        output = model(inputs)
        if isinstance(output, tuple):
            output, _ = output

        test_loss += criterion(output, targets).data.item()
        pred = get_predictions(output)
        test_error += error(pred, targets.data)

        batch_avg_loss = test_loss / (1 + idx)
        batch_avg_err = test_error / (1 + idx)
        if logger is not None:
            logger.log('val', epoch, idx+1, n_batches, batch_avg_loss, batch_avg_err, verbose=True)

    test_loss /= n_batches
    test_error /= n_batches
    return test_loss, test_error


def adjust_learning_rate(lr, decay, optimizer, cur_epoch, every_n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // every_n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for input, target in input_loader:
        data = Variable(input.cuda(), volatile=True)
        label = Variable(target.cuda())
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input,target,pred])
    return predictions

