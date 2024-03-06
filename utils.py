import os
import sys
import time

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.utils import _log_api_usage_once

class EEG(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
    
        if self.train:
            self.train_data = np.load(os.path.join(self.root, "X_train_valid.npy"))
            self.train_labels = np.load(os.path.join(self.root, "y_train_valid.npy"))
        else:
            self.test_data = np.load(os.path.join(self.root, "X_test.npy"))
            self.test_labels = np.load(os.path.join(self.root, "y_test.npy"))

    def __getitem__(self, index):
        if self.train:
            series, target = self.train_data[index], self.train_labels[index]
        else:
            series, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            series = self.transform(series)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return series, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        
class ToTensor:
    def __init__(self) -> None:
        _log_api_usage_once(self)

    def __call__(self, pic):
        return torch.from_numpy(pic)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

TOTAL_BAR_LENGTH = 20.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    L.append(' %d/%d ' % (current+1, total))

    msg = ''.join(L)
    sys.stdout.write(msg)

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()