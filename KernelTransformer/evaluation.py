
import sys
import time

import numpy
import torch

from cfg import Config


def run_validation(model, dataloader):
    model.eval()
    errs = []
    duration = 0
    for i, (srcs, dsts) in enumerate(dataloader):
        if Config["is_cuda"]:
            srcs = srcs.cuda()
            dsts = dsts.cuda()

        start = time.time()
        with torch.no_grad():
            preds = model(srcs)
        end = time.time()
        elapsed = end - start
        duration += elapsed

        diff = dsts.data - preds.data
        err = diff ** 2
        err = err.mean(dim=3).mean(dim=1).cpu().numpy()
        errs.append(err)
    sys.stdout.write("Total time: {}\n".format(duration))
    errs = numpy.vstack(errs)
    err = numpy.sqrt(errs.mean())
    sys.stdout.write("validation = {:.3f}\n".format(err))
    sys.stdout.flush()
    model.train()
    return errs

def row_errors(errs):
    if len(errs.shape) == 3:
        n_validation, n_out, H = errs.shape
        for i in xrange(H):
            row_err = errs[:,:,i]
            err = numpy.sqrt(row_err.mean())
            sys.stdout.write("Row {0}: {1:.3f}\n".format(i, err))
    elif len(errs.shape) == 2:
        n_validation, H = errs.shape
        for i in xrange(H):
            row_err = errs[:,i]
            err = numpy.sqrt(row_err.mean())
            sys.stdout.write("Row {0}: {1:.3f}\n".format(i, err))
    else:
        raise ValueError("Incorrect error shape.")

