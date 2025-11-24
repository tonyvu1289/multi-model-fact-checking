import math
import datetime
import os
import random
import re

from torch import optim
from read_data import get_dataset
import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import f1_score, confusion_matrix
import torch.nn.functional as F
import copy
from tqdm import tqdm, trange

from model import *


def one_hot(a, num_classes):
    v = np.zeros(num_classes, dtype=int)
    v[a] = 1
    return v


def clean_data(text):
    if str(text) == 'nan':
        return text
    text = re.sub("(<p>|</p>|@)+", '', text)
    return text.strip()


def encode_one_sample(sample):
    claim = sample[0]
    text_evidence = sample[1]
    image_evidence = sample[2]
    label = sample[3]
    claim_id = sample[4]

    label2idx = {
        'refuted': 2,
        'NEI': 1,
        'supported': 0
    }

    encoded_sample = {}
    encoded_sample["claim_id"] = claim_id
    encoded_sample["claim"] = claim
    encoded_sample["label"] = torch.tensor(one_hot(label2idx[label], 3), dtype=float)
    encoded_sample['text_evidence'] = [clean_data(t) for t in text_evidence]
    encoded_sample['image_evidence'] = image_evidence.tolist()

    return encoded_sample


class ClaimVerificationDataset(torch.utils.data.Dataset):
    def __init__(self, claim_verification_data):
        self._data = claim_verification_data
        # self._processor = processor

        self._encoded = []
        for d in self._data:
            self._encoded.append(encode_one_sample(d))

    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]

    def to_list(self):
        return self._encoded


def make_batch(train_data, batch_size=128, shuffle=True):
    claim_ids = []
    claim_labels = []
    claim_features = []

    if shuffle:
        train_data = train_data.to_list() if not isinstance(train_data, list) else train_data
        random.shuffle(train_data)

    for d in train_data:
        claim_ids.append(d['claim_id'])
        claim_labels.append(d['label'])
        claim_features.append(d)

    num_batches = math.ceil(len(train_data) / batch_size)
    train_features_batch = [claim_features[batch_size * y:batch_size * (y + 1)] for y in range(num_batches)]
    # train_label_batch = [torch.cat(claim_labels[batch_size * y: batch_size * (y + 1)], out=torch.Tensor(len(claim_labels[batch_size * y:batch_size * (y + 1)]), 1, 3).to(device)) for y in range(num_batches)]
    train_label_batch = [claim_labels[batch_size * y: batch_size * (y + 1)] for y in range(num_batches)]
    train_id_batch = [claim_ids[batch_size * y:batch_size * (y + 1)] for y in range(num_batches)]

    return train_features_batch, train_label_batch, train_id_batch


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs.squeeze(), targets.float())
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss


def train_model(train_data, batch_size, epoch=1, is_val=False, val_data=None, claim_pt="roberta-base", vision_pt='vit',
                long_pt="longformer", device=None):
    # if n_gpu:
    #     device = torch.device('cuda:{}'.format(n_gpu) if torch.cuda.is_available() else 'cpu')
    # else:
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalClassification(device, claim_pt, vision_pt, long_pt)
    # model = MultiModalClassificationNoAttention(device, claim_pt, vision_pt, long_pt)
    model = model.to(device)

    # multi-gpu support: wrap model with DataParallel if multiple GPUs available
    num_gpus = torch.cuda.device_count()
    use_dataparallel = num_gpus > 1
    if use_dataparallel:
        model = nn.DataParallel(model)
    # create optimizer after wrapping so it references correct parameters
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    loss_function = FocalLoss(gamma=2).to(device)
    loss_vals = []

    print('Training.......')
    best_model = model
    best_acc = 0

    chk_dir = "model_dump/model_verification_{}_{}_{}_{}".format(
        str(claim_pt),
        str(long_pt),
        str(vision_pt),
        str(datetime.datetime.now().strftime("%d-%m_%H-%M"))
    )
    os.makedirs(chk_dir)
    os.makedirs("{}/checkpoint".format(chk_dir))

    X, y, _ = make_batch(train_data, batch_size=batch_size)
    for e in trange(epoch):
        model.train()
        total_loss = 0
        print("Epoch {}:\n ".format(e + 1))

        for i in trange(len(X)):
            optimizer.zero_grad()
            batch_x = X[i]
            # call model to get scores only; build label tensor separately so shapes are consistent across GPUs
            score = model(batch_x)
            lb = torch.stack(y[i]).to(device)
            loss = loss_function(score.to(device), lb)
            loss.backward()

            optimizer.step()
            total_loss = total_loss + loss.item()

        loss_vals.append(total_loss)
        print("Loss: {}\n".format(total_loss))

        if is_val and val_data:
            truelb, predlb, _ = predict(val_data, model, batch_size=batch_size)
            mif1 = f1_score(truelb, predlb, average='micro')
            if mif1 > best_acc:
                best_acc = copy.deepcopy(mif1)
                # keep best_model as the underlying module (unwrapped) for portability
                best_model = copy.deepcopy(model.module if isinstance(model, nn.DataParallel) else model)
            print("Macro F1-score: {}\n".format(f1_score(truelb, predlb, average='macro')))
            print("F1-score: {}\n".format(f1_score(truelb, predlb, average='micro')))
        else:
            best_model = copy.deepcopy(model.module if isinstance(model, nn.DataParallel) else model)
        print('===========\n')

        # Save checkpoint with state_dict of the underlying module so keys are consistent
        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({
            'total_epochs': epoch,
            'current_epoch': e,
            'batch_size': batch_size,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict()
        }, "{}/checkpoint/checkpoint_{}.pt".format(
            str(chk_dir),
            str(e))
        )

    # Save best model's state_dict (unwrapped)
    best_state = best_model.state_dict() if not isinstance(best_model, nn.DataParallel) else best_model.module.state_dict()
    torch.save({'model_state_dict': best_state}, '{}/best_model.pt'.format(chk_dir))

    return best_model, loss_vals, claim_pt


def train_resume(train_data, chkpoint, is_val=False, val_data=None, claim_pt="roberta-base",
                 vision_pt='vit', long_pt="longformer", device=None):
    # create base model, load checkpoint (handle potential 'module.' prefix), then wrap for multi-gpu
    def _strip_module_prefix(state_dict):
        # If the keys contain 'module.' prefix (from DataParallel), strip it
        if any(k.startswith('module.') for k in state_dict.keys()):
            return {k.replace('module.', ''): v for k, v in state_dict.items()}
        return state_dict

    model = MultiModalClassification(device, claim_pt, vision_pt, long_pt)
    state = chkpoint['model_state_dict']
    state = _strip_module_prefix(state)
    model.load_state_dict(state)
    model = model.to(device)

    # wrap for multi-gpu if available
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
    print(model)
    loss_function = FocalLoss(gamma=2)
    loss_function = loss_function.to(device)
    epoch = chkpoint['total_epochs']
    batch_size = chkpoint['batch_size']

    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    optimizer.load_state_dict(chkpoint['optimizer_state_dict'])
    loss_vals = []

    print('Training.......')
    best_model = model
    best_acc = 0
    X, y, _ = make_batch(train_data, batch_size=batch_size)

    chk_dir = "model_dump/model_verification_{}_{}_{}_{}".format(
        str(claim_pt),
        str(long_pt),
        str(vision_pt),
        str(datetime.datetime.now().strftime("%d-%m_%H-%M"))
    )
    os.makedirs(chk_dir)
    os.makedirs("{}/checkpoint".format(chk_dir))

    for e in trange(chkpoint['current_epoch'], epoch):
        model.train()
        total_loss = 0
        print("Epoch {}:\n ".format(e + 1))

        for i in trange(len(X)):
            optimizer.zero_grad()
            batch_x = X[i]
            # call model to get scores only; build label tensor separately so shapes are consistent across GPUs
            score = model(batch_x)
            lb = torch.stack(y[i]).to(device)
            loss = loss_function(score.to(device), lb)
            loss.backward()

            optimizer.step()
            total_loss = total_loss + loss.item()

        loss_vals.append(total_loss)
        print("Loss: {}\n".format(total_loss))

        if is_val and val_data:
            truelb, predlb, _ = predict(val_data, model, batch_size=batch_size)
            mif1 = f1_score(truelb, predlb, average='micro')
            if mif1 > best_acc:
                best_acc = copy.deepcopy(mif1)
                best_model = copy.deepcopy(model)
            print("Macro F1-score: {}\n".format(f1_score(truelb, predlb, average='macro')))
            print("F1-score: {}\n".format(f1_score(truelb, predlb, average='micro')))
        else:
            best_model = copy.deepcopy(model)
        print('===========\n')

        model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({
            'total_epochs': epoch,
            'current_epoch': e,
            'batch_size': batch_size,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict()
        }, "{}/checkpoint/checkpoint_{}.pt".format(
            str(chk_dir),
            str(e))
        )

    torch.save(best_model, '{}/best_model.pt'.format(chk_dir))

    return best_model, loss_vals, claim_pt


def predict(test_data, model, batch_size, device=None):
    # model = nn.DataParallel(model)
    # move to device and wrap for multi-gpu if not already wrapped
    model = model.to(device)
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)

    ground_truth = []
    predicts = []
    ids = []

    X, y, z = make_batch(test_data, batch_size=batch_size, shuffle=False)

    model.eval()
    print('Predict.......')

    for i in trange(len(X)):
        batch_x = X[i]
        batch_y = y[i]
        batch_z = z[i]
        # model returns scores only; no label expected from forward
        scores = model(batch_x)
        scores = scores.reshape(-1, 3)

        if not ids:
            ids = [i for i in batch_z]
        else:
            ids.extend([i for i in batch_z])

        if not ground_truth:
            ground_truth = [np.argmax(label.tolist(), -1) for label in batch_y]
        else:
            ground_truth.extend([np.argmax(label.tolist(), -1) for label in batch_y])

        if not predicts:
            predicts = [np.argmax(score.tolist(), -1) for score in scores]
        else:
            predicts.extend([np.argmax(score.tolist(), -1) for score in scores])

    return ground_truth, predicts, ids


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train, val, test = get_dataset('../data')
    dev_claim = ClaimVerificationDataset(val)
    train_claim = ClaimVerificationDataset(train)
    test_claim = ClaimVerificationDataset(test)

    model, loss, _ = train_model(train_claim[0:10], batch_size=5, epoch=5, is_val=True, val_data=dev_claim[1:10], device=device)
    # torch.save(model, '../output/claim_verification.pt')

    gt, prd, ids = predict(test_claim[1:10], model, 16)
    print("Test result micro: {}\n".format(f1_score(gt, prd, average='micro')))
    print("Test result macro: {}\n".format(f1_score(gt, prd, average='macro')))
    print(confusion_matrix(gt, prd))
