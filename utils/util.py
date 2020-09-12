import os
import torch


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


def save_model(model, optimizer, path, step):
    check_path = os.path.join(path, "step_%d" % step)
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step},
               check_path)


def load_model(path, model, optimizer):
    check_point = torch.load(path, map_location='cuda')
    model.load_state_dict(check_point['model'])
    optimizer.load_state_dict(check_point['optimizer'])
    step = check_point['step']
    return model, optimizer, step