
import torch
import torch.optim as optim

from torch.autograd import Variable

from cfg import Config

def enable_gpu(model, gpu=0):
    if gpu is not None and gpu < 0:
        Config["is_cuda"] = False
        return model
    Config["is_cuda"] = True
    torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.enabled = False
    if torch.cuda.device_count() > 1:
        if gpu < torch.cuda.device_count():
            torch.cuda.set_device(gpu)
    model.cuda()
    return model

def create_variable(size):
    if Config["is_cuda"]:
        variable = Variable(torch.cuda.FloatTensor(*size))
    else:
        variable = Variable(torch.FloatTensor(*size))
    return variable

def build_optimizer(model, decay=10, base_lr=0.01, update="transform"):
    model.update_group(update)
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        params.append(param)
    optimizer = optim.Adam(params, lr=base_lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay)
    return optimizer, scheduler

