import os
import argparse
import time
import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from timm.data import Mixup
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from utils import EEG, ToTensor
from engine import train, test

def parse_option():

    parser = argparse.ArgumentParser('EEG Models arguments')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epoch5s')

    # optimization
    parser.add_argument('--learning_rate_fc', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model
    parser.add_argument('--net', type=str, required=True,
                        help='choose pre-trained model')

    # dataset
    parser.add_argument('--root', type=str, default='./data',
                        help='dataset')
    
    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=2,
                        help='number of trials')
    parser.add_argument('--gpu', type=str, default='0',
                        help='gpu to use')

    args = parser.parse_args()

    args.filename = '{}_{}_lrf_{}_decay_{}_bsz_{}_epochs_{}_trial_{}'. \
        format(args.dataset, args.net,
            args.learning_rate_fc, args.weight_decay, args.batch_size, args.epochs, args.trial)

    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    return args

def main():

    args = parse_option()

    # environment settings
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # data preparation
    print('==> Preparing data..')
    
    train_set = EEG(args.root, transform=ToTensor())
    train_loader = DataLoader(train_set, shuffle=True, num_workers=args.num_workers, batch_size=args.batch_size)

    test_set = EEG(args.root, transform=ToTensor())
    test_loader = DataLoader(test_set, shuffle=False, num_workers=args.num_workers, batch_size=args.batch_size)

    # create model
    print('==> Building model..')

    model = get_network(args.net, classes, device)

    # data augmentation
    mixup = 0.8
    cutmix = 1.0
    cutmix_minmax = None
    mixup_prob = 1.0
    mixup_switch_prob = 0.5
    mixup_mode = 'batch'
    smoothing = 0.1

    mixup_fn = None
    mixup_active = mixup > 0 or cutmix > 0. or cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=mixup, cutmix_alpha=cutmix, cutmix_minmax=cutmix_minmax,
            prob=mixup_prob, switch_prob=mixup_switch_prob, mode=mixup_mode,
            label_smoothing=smoothing, num_classes=classes)

    # criterion
    if mixup_active:
        criterion_train = SoftTargetCrossEntropy()
    elif smoothing:
        criterion_train = LabelSmoothingCrossEntropy(smoothing)
    else:
        criterion_train = torch.nn.CrossEntropyLoss()

    criterion_test = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate_fc, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # scheduler
    args.warmup = 3*len(train_loader)
    scheduler = CosineLRScheduler(optimizer, t_initial=len(train_loader)*args.epochs, warmup_t=args.warmup)
    
    # tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join('runs', time.strftime(f"%Y-%m-%d {time.localtime().tm_hour+8}:%M:%S", time.localtime())))
        
    # training loop
    best_acc = 0.0
    for epoch in range(args.epochs):

        print('\nEpoch: %d' % (epoch+1))

        # train model
        
        acc_train = train(model, train_loader, mixup_fn, optimizer, scheduler, epoch, criterion_train, device, Split, target)
                
        # test model
        with torch.no_grad():
            acc_test = test(model, val_loader, criterion_test, device, Split, target)

       # save model
        acc = acc_test
        if best_acc < acc:
            filename_sub = 'target:{tar}_acc:{best_acc}.pth'.format(tar=args.target, best_acc=format(best_acc, '.6f'))
            filename_best = 'target:{tar}_acc:{acc}.pth'.format(tar=args.target, acc=format(acc, '.6f'))
            sub_path = os.path.join(args.model_folder, filename_sub)
            best_path = os.path.join(args.model_folder, filename_best)

            if best_acc != 0:
                    os.remove(sub_path)
            torch.save(model.state_dict(), best_path)
            best_acc = acc
        
        writer.add_scalar('Train/Accuracy', acc_train , epoch)
        writer.add_scalar('Test/Accuracy', acc_test , epoch)

    writer.close()
    print("Done!")

if __name__ == '__main__':
    main()