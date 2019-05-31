#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import argparse
import numpy as np
import threading
import torch
import torch.nn as nn

from time import gmtime, strftime

from KernelTransformer.cfg import Config
from KernelTransformer.cfg import LAYERS
from KernelTransformer.cfg import MODEL_DIR
from KernelTransformer.evaluation import row_errors
from KernelTransformer.evaluation import run_validation
from KernelTransformer.Loader.data_loader import prepare_dataset
from KernelTransformer.Loader.model_loader import build_ktnconv
from KernelTransformer.util import build_optimizer
from KernelTransformer.util import enable_gpu


def run_steps(ktnconv, optimizer, dataloader, steps):
    Loss = nn.MSELoss()
    tied_weights = ktnconv.tied_weights
    n_transform = ktnconv.n_transform
    n_splits = ktnconv.n_transform

    for step in xrange(steps):
        losses = []
        display = len(dataloader)
        for i, (srcs, dsts) in enumerate(dataloader):
            if Config["is_cuda"]:
                srcs = srcs.cuda()
                dsts = dsts.cuda()
            rows = np.random.permutation(n_transform).reshape((n_splits, -1))
            split_rows = rows.astype(int).tolist()

            optimizer.zero_grad()
            for j, rows in enumerate(split_rows):
                targets = []
                t = threading.Thread(target=split_target, args=(dsts, rows, tied_weights, targets))
                t.start()
                pred = ktnconv(srcs, rows)
                t.join()
                targets = targets[0]
                loss = Loss(pred, targets)
                loss.backward()
                loss = loss.item()
                losses.append(loss)
            optimizer.step()

            # display progress
            if len(losses) == display * n_splits:
                display_progress(i, losses)
                losses = []
        if len(losses) > 0:
            display_progress(i, losses)
    return ktnconv

def split_target(dsts, rows, tied_weights, results):
    sub_dsts = []
    for row in rows:
        t = row * tied_weights
        b = t + tied_weights
        sub_dst = dsts[:,:,t:b,:]
        sub_dsts.append(sub_dst)
    sub_dsts = torch.cat(sub_dsts, dim=2)
    results.append(sub_dsts)

def display_progress(iteration, losses):
    losses = np.array(losses).mean()
    sys.stdout.write("Iteration {0:3d}: loss = {1:.3f}, {2}\n".format(iteration+1,
                                                                      np.sqrt(losses),
                                                                      strftime("%H:%M:%S", gmtime())))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--source', choices=["pascal", "imagenet", "coco"], default='pascal')
    parser.add_argument('--update', choices=["transform", "kernel", "all"], default='transform')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('layer', choices=LAYERS)
    args = parser.parse_args()

    # Check if output model exists.
    output = os.path.join(MODEL_DIR,
                          "{0}{1}.{2}.pt".format(args.source,
                                                 args.layer,
                                                 args.update))
    if os.path.isfile(output):
        sys.stderr.write("Model {} exists.\n".format(output))
        return

    # Create dataloader
    train_loader, valid_loader = prepare_dataset(args.layer,
                                                 src_cnn=args.source,
                                                 batch_size=args.batch)

    # Initialize the model
    ktnconv = build_ktnconv(args.layer, network=args.source)
    if torch.cuda.is_available():
        sys.stderr.write("Enable GPU\n")
        ktnconv = enable_gpu(ktnconv, gpu=args.gpu)

    # Initialize optimizer
    epochs = 8
    steps = 5
    decay = epochs / 2
    optimizer, scheduler = build_optimizer(ktnconv,
                                           decay=decay,
                                           base_lr=args.lr,
                                           update=args.update)

    run_validation(ktnconv, valid_loader)
    for epoch in xrange(epochs):
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        sys.stdout.write("Epoch {0}: learning rate = {1}, {2}\n".format(epoch+1,
                                                                        lr,
                                                                        strftime("%H:%M:%S", gmtime())))
        ktnconv = run_steps(ktnconv, optimizer, train_loader, steps)
        # run validation
        diffs = run_validation(ktnconv, valid_loader)
    row_errors(diffs)
    ktnconv.cpu()
    torch.save(ktnconv.state_dict(), output)

if __name__ == "__main__":
    main()

