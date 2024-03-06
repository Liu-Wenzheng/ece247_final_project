import torch
from utils import progress_bar, labeltranslation, fc_input
import time

def train(model, train_loader, mixup_fn, optimizer, scheduler, epoch, criterion, device, Split, tar):
    model.train()

    train_loss = 0.0
    correct = 0.0
    total = 0.0

    num_batches_per_epoch = len(train_loader)
    
    print(optimizer.param_groups[0]['lr'])

    for batch_index, (images, labels) in enumerate(train_loader):

        labels = labeltranslation(labels, Split, tar)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + batch_index
        scheduler.step(step)

        images = images.to(device)
        labels = labels.to(device)

        # data augmentation
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        # forward propogation
        pred = model(images)
        loss = criterion(pred, labels)
        
        # backward propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # result statistics
        train_loss += loss.item()
        total += labels.size(0)
        _, predicted = pred.max(1)
        if mixup_fn is not None:
            _, target = labels.max(1)
            correct += predicted.eq(target).sum().item()
        else:
            correct += predicted.eq(labels).sum().item()
        
        progress_bar(batch_index, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_index+1), 100.*correct/total, correct, total))

    return correct/len(train_loader.dataset)

def test(model, val_loader, criterion, device, Split, tar):
    model.eval()

    test_loss = 0.0
    correct = 0.0
    total = 0.0

    for batch_index, (images, labels) in enumerate(val_loader):
        labels = labeltranslation(labels, Split, tar)

        images = images.to(device)
        labels = labels.to(device)

        # forward propogation
        outputs = model(images)
        loss = criterion(outputs, labels)

        # result statistics
        test_loss += loss.item()
        total += labels.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        progress_bar(batch_index, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_index+1), 100.*correct/total, correct, total))
    
    return correct/len(val_loader.dataset)

def train_kd(model, modelT, T, alpha, train_loader, mixup_fn, device, scheduler, epoch, optimizer, criterion, Split, tar, KD_LOSS, Scheduler, Optimizer):
    model.train()

    train_loss = 0.0
    correct = 0.0
    total = 0.0

    num = len(T)
    num_batches_per_epoch = len(train_loader)
        
    for batch_index, (images, labels) in enumerate(train_loader):
        labels = labeltranslation(labels, Split, tar)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + batch_index
        scheduler.step(step)
        for s in Scheduler:
            s.step(epoch)

        images = images.to(device)
        labels = labels.to(device)

        # data augmentation
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        # add hooks to student model
        FC = [fc_input(model) for i in range(num)]
        for i in range(num):
            FC[i].fc_input_data() 

        # student model forward propogation
        pred = model(images.requires_grad_())

        # extract pre_fc layer features
        Feature = [FC[i].fc_input() for i in range(num)]

        # teacher models forward propogation
        with torch.no_grad():
                predT = [m(images) for m in modelT]

        # distillation related loss calculation
        LOSST = 0
        L = torch.nn.LogSoftmax(dim=1)

        for i in range(num):
            lossg = KD_LOSS[i].forward(Feature[i], predT[i], T[i])
            LOSST += lossg*alpha[i]

        if mixup_fn is not None:
            loss = criterion(L(pred), labels)
        else:
            loss = criterion(pred, labels)

        loss = loss * (1 - sum(alpha)) + LOSST

        # student and teacher models backward propogation
        optimizer.zero_grad()
        for i in range(num):
            Optimizer[i].zero_grad()

        loss.backward()

        optimizer.step()
        for i in range(num):
            Optimizer[i].step()

        # result statistics
        train_loss += loss.item()
        total += labels.size(0)
        _, predicted = pred.max(1)
        if mixup_fn is not None:
            _, target = labels.max(1)
            correct += predicted.eq(target).sum().item()
        else:
            correct += predicted.eq(labels).sum().item()
        
        progress_bar(batch_index, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_index+1), 100.*correct/total, correct, total))

    return correct/len(train_loader.dataset)

