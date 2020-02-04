import torch
# from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
# import torchvision
import torchvision.transforms as transforms
from models import MTANet
from dataset import Aff2
import myloss
import numpy as np
import argparse
import os
from torch.utils.data import DataLoader, RandomSampler
import shutil
import time
import logging
import math
import sklearn.metrics as sm
from models.metrics import ArcMarginProduct
from utils import prep_experiment, print_eval


parser = argparse.ArgumentParser(description='MTANet Training')
parser.add_argument('-a', '--arch', type=str, default='resnext', metavar='ARCH', help='Using model')
parser.add_argument('--bs', type=int, default=80, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=100, help='Epoch')
parser.add_argument('--workers', type=int, default=0, help='Dataloader num_worker')
parser.add_argument('--poly_exp', type=float, default=1.0, help='Polynomial LR exponent')
parser.add_argument('--resume', type=str, default='', metavar='PATH', help='Path to checkpoint')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='Manual epoch number')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Evaluate mode on validation set')
parser.add_argument('-p', '--print_freq', type=int, default=10, metavar='N', help='Print frequency')
parser.add_argument('--ckpt_path', type=str, default='logs/ckpt')
parser.add_argument('--tb_path', type=str, default='logs/tb')
parser.add_argument('--exp', type=str, default='exp', help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='', help='add tag to tb dir')
parser.add_argument('--loss_type', type=str, default='learned', help='fixed or learned')
parser.add_argument('-arc', '--arcface', dest='arcface', action='store_true', help='Using acrface metric')
parser.add_argument('--optim', type=str, default='sgd', help='SGD or Adam')
parser.add_argument('-smp', '--sampler', dest='sampler', action='store_true', help='Using sampler')
args = parser.parse_args()
args.best_record = {'epoch': -1, 'val_loss': 1e10, 'best_va': 1e10, 'best_acc1': 0,
                    'best_au_strict': 0, 'best_expr_f1': 0, 'best_au_f1': 0}

size = (112, 112)     # for VGGFace, not 2  (96, 96)
initial_ses = [1., 1., 1.]
# path = '2DFer_' + args.model + "_" + str(args.bs)
# lr_decay_start = 50  # 50
# lr_decay_every = 5
# lr_decay_rate = 0.9
cudnn.benchmark = True
device = torch.device('cuda')

# if args.arch == 'v1':
#     # net = ResNet34()
#     # net = Aff2net.aff2net(num_classes=17, include_top=True)
#     net = Aff2net_nopretrain.aff2net(num_classes=17)
#     net = net.to(device)
# else:  # opt.model == 'ResNet50'
# net = Aff2netv2.aff2net(initial_ses=initial_ses, arc_face=args.arcface, backbone=args.arch)
net = MTANet.aff2net(initial_ses=initial_ses, arc_face=args.arcface, backbone=args.arch)
net = net.to(device)
# raise ValueError("Invalid model")

if args.arcface:
    metric_fc = ArcMarginProduct(1024, 7, s=30, m=0.5, easy_margin=False)
    metric_fc.to(device)


def main():
    # same implementation in net class
    # for name, value in net.named_parameters():
    #     if str(name).split('fc')[0] != '':
    #         value.requires_grad = False
        #print(value.requires_grad)
    # initialize best values
    best_acc1 = 0
    best_va = 0
    best_au_strict = 0
    # best_au_soft = 0
    best_loss = 1e10
    best_expr_f1 = 0
    best_au_f1 = 0
    final_cm = 0
    final_mcm = 0
    # setup optimizer
    params = filter(lambda p: p.requires_grad, net.parameters())
    # for param in params:
    #     print(type(param.data), param.size())
    if args.arcface:
        params = [{'params': params}, {'params': metric_fc.parameters()}]
    if args.optim == 'sgd':
        # [{'params': params}, {'params': metric_fc.parameters()}]
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    else:
        # [{'params': params}, {'params': metric_fc.parameters()}]
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(params, lr=opt.lr, weight_decay=args.weight_decay, amsgrad=)
    # if args.lr_schedule:
    poly_lambda = lambda epoch: math.pow(1 - epoch / args.epochs, args.poly_exp)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_lambda)
    criterion = myloss.MultiTaskLoss(loss_type=args.loss_type, loss_uncertainties=net.get_loss_weights(), gamma=2)

    # if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # args.start_epoch = checkpoint['epoch']  # if adding start_epoch to args
        # TODO: loading best results
        best_acc1 = checkpoint['best_acc1']
        best_loss = checkpoint['best_loss']
        best_va = checkpoint['best_va']
        best_au_strict = checkpoint['bse_au_strict']
        best_expr_f1 = checkpoint['best_expr_f1']
        best_au_f1 = checkpoint['best_au_f1']
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading
    # TODO: check using whether vggface mean_bgr or my dataset mean_bgr
    # mean_bgr = np.array([91.4953, 103.8827, 131.0912])      # from VGGFace2 resnet50_ft.prototxt
    transform_train = transforms.Compose([
        transforms.Resize(size),        # follow VGGface's input 224*224 3C RGB
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),                  # range [0, 255] -> [0.0, 1.0]
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]),  # range [0.0, 1.0] -> [-1.0, 1.0]
    ])
    transform_val = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5]),  # range [0.0, 1.0] -> [-1.0, 1.0]
    ])
    # dataset label=[v:(float), a:(float), AUs:one hot code-len=8, Expr:int]
    # train_set = TestData(transform=transform_train)     # , flag="train"
    # val_set = TestData(transform=transform_val)     # , flag="val"
    train_set = Aff2(transform=transform_train, flag="train")
    val_set = Aff2(transform=transform_val, flag="val")
    if args.sampler:
        train_sampler = RandomSampler(train_set, True, int(1e5))
        val_sampler = RandomSampler(val_set, True, int(2e4))
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=shuffle, sampler=train_sampler,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=shuffle, sampler=val_sampler,
                            num_workers=args.workers, pin_memory=True)

    writer = prep_experiment(args)

    if args.evaluate:
        loss, loss_va, top1, au_strict, cm, mcm, expr_f1, au_f1 = validate(val_loader, net, criterion, 0, args, writer)
        args.best_record['val_loss'] = loss.avg
        args.best_record['best_acc1'] = top1.avg
        args.best_record['best_va'] = loss_va.avg
        args.best_record['best_au_strict'] = au_strict.avg
        args.best_record['best_expr_f1'] = expr_f1.avg
        args.best_record['best_au_f1'] = au_f1.avg
        print_eval(args)
        np.save(os.path.join(args.exp_path, "CM.npy"), np.array(final_cm))
        np.save(os.path.join(args.exp_path, "MCM.npy"), np.array(final_mcm))
        return

    for epoch in range(args.start_epoch, args.epochs):
        train(train_loader, net, criterion, optimizer, epoch, args, writer)

        # evaluate on validation set
        loss, loss_va, top1, au_strict, cm, mcm, expr_f1, au_f1 = validate(val_loader, net, criterion, epoch, args, writer)

        # if args.lr_schedule:
        scheduler.step(epoch)
        # else:
        #     adjust_learning_rate(optimizer, epoch, args)
        # train one epoch

        # TODO: best acc with VA, AU and Expr or separately
        # remember best acc@1 and save checkpoint
        is_best = ((loss_va.avg > best_va) & (loss.avg < best_loss)) | \
                  ((au_f1.avg > best_au_f1) & (expr_f1.avg > best_expr_f1))
        best_loss = min(loss.avg, best_loss)
        best_acc1 = max(top1.avg, best_acc1)
        best_va = max(loss_va.avg, best_va)
        best_au_strict = max(au_strict.avg, best_au_strict)
        # best_au_soft = max(au_soft.avg, best_au_soft)
        best_expr_f1 = max(expr_f1.avg, best_expr_f1)
        best_au_f1 = max(au_f1.avg, best_au_f1)
        final_cm += cm
        final_mcm += mcm
        if is_best:
            logging.info("Got best model for now. Saving model...")
            args.best_record['epoch'] = epoch + 1
            args.best_record['val_loss'] = best_loss
            args.best_record['best_acc1'] = best_acc1
            args.best_record['best_va'] = best_va
            args.best_record['best_au_strict'] = best_au_strict
            # args.best_record['best_au_soft'] = best_au_soft
            args.best_record['best_expr_f1'] = best_expr_f1
            args.best_record['best_au_f1'] = best_au_f1
            print_eval(args)
            np.save(os.path.join(args.exp_path, "CM.npy"), np.array(final_cm))
            np.save(os.path.join(args.exp_path, "MCM.npy"), np.array(final_mcm))

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net.state_dict(),
            'best_loss': best_loss,
            'best_acc1': best_acc1,
            'best_va': best_va,
            'bse_au_strict': best_au_strict,
            'best_expr_f1': best_expr_f1,
            'best_au_f1': best_au_f1,
            'optimizer': optimizer.state_dict(),
        }, is_best, epoch + 1)


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Total Loss', ':.4f')
    loss_va = AverageMeter('VA Loss', ':.4f')
    loss_au = AverageMeter('AU Loss', ':.4f')
    loss_expr = AverageMeter('Expr Loss', ':.4f')
    top1 = AverageMeter('Expr Acc@1', ':6.2f')
    v_ccc = AverageMeter('V CCC', ':6.2f')
    a_ccc = AverageMeter('A CCC', ':6.2f')
    # top3 = AverageMeter('Expr Acc@3', ':6.2f')
    au_strict = AverageMeter('AU Strict Acc', ':6.2f')
    # au_soft = AverageMeter('AU Soft Acc', ':6.2f')
    # au_category = AverageMeter('AU Cgr Acc', '')
    f1_expr = AverageMeter('Expr F1-score', ':6.2f')
    f1_aus = AverageMeter('AU F1-score', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, loss_va, loss_au, loss_expr, v_ccc, a_ccc, top1, f1_expr, au_strict, f1_aus],    # , au_category
        prefix="Train Epoch: [{}]".format(epoch + 1)
    )
    # switch to training mode
    curr_iter = epoch * len(train_loader)
    model.train()

    end = time.time()
    for batch_index, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # print("image size is ", images.size())
        # print("label type is ", type(labels))
        images = images.to(device)      # , torch.tensor(labels).to(device)
        labels = labels.to(device)

        v_labels = labels[:, 0]
        a_labels = labels[:, 1]
        # if aus labels store separately in labels list: labels[2:-1]
        aus_labels = labels[:, 2:-1]
        expr_labels = labels[:, -1]
        # target = (v_labels, a_labels, aus_labels, expr_labels)
        # print(expr_labels.long())
        # print("v label type is ", v_labels.type())
        # print("v label size is ", v_labels.size())
        # compute output
        outputs = model(images)
        # print("output:", outputs)
        # print("train P_outputs:", P_outputs)outputs
        # print("v pred size is ", outputs[:, 0].size())
        # print("v pred type is ", outputs[:, 0].type())
        # if args.arch == "v2":
        va_output, aus_output, expr_output = outputs
        # else:
        # v_output, a_output, aus_output, expr_output = outputs[:, 0], outputs[:, 1], outputs[:, 2:10], outputs[:, 10:]
        # print(expr_output)
        if args.arcface:
            expr_output, expr_labels = metric_fc(expr_output, expr_labels)
            outputs = (va_output, aus_output, expr_output)

        loss, va_bs, au_bs, (va_loss, v_loss, a_loss, au_loss, expr_loss) = \
            criterion(outputs, v_labels, a_labels, aus_labels, expr_labels)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        batch_size = images.size(0)
        (acc1, acc3), expr_bs, expr_f1 = expr_accuracy(expr_output, expr_labels, topk=(1, 3))
        losses.update(loss.item(), batch_size)
        loss_va.update(va_loss.item(), va_bs)
        loss_au.update(au_loss.item(), au_bs)
        loss_expr.update(expr_loss.item(), expr_bs)
        top1.update(acc1[0], expr_bs)
        v_ccc.update(v_loss, va_bs)
        a_ccc.update(a_loss, va_bs)
        # top3.update(acc3[0], expr_bs)
        strict_acc, au_f1 = aus_accuracy(aus_output, aus_labels, threshold=0.5)
        au_strict.update(strict_acc, au_bs)
        # au_soft.update(soft_acc, au_bs)
        # au_category.update(cgr_acc, au_bs)
        f1_expr.update(expr_f1, expr_bs)
        f1_aus.update(au_f1, au_bs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        curr_iter += 1
        lr = optimizer.state_dict()["param_groups"][-1]['lr']
        
        w1, w2, w3 = model.get_loss_weights()
        if batch_index % args.print_freq == 0 or batch_index == len(train_loader) - 1:
            weight_info = "\tw1={:0.4f}\tw2={:0.4f}\tw3={:0.4f}".format(w1.item(), w2.item(), w3.item())
            info = progress.info(batch_index + 1) + "\tlr={}".format(lr) + weight_info

            # msg = '[epoch {}], [iter {} / {}], [train main loss {:0.6f}], [VA loss {:0.6f}], [AUs loss {:0.6f}],' \
            #       '[Expr loss {:0.6f}], [lr {:0.6f}]'.format(
            #         epoch, batch_index + 1, len(train_loader), losses.avg, loss_va.avg, loss_au.avg, loss_expr.avg,
            #         lr)   # optimizer.param_groups[-1]['lr']

            logging.info(info)

        # Log tensorboard metrics for each iteration of the training phase
        writer.add_scalar('training/loss', losses.val, curr_iter)   # (losses.val)
        writer.add_scalar('training/lr', lr, curr_iter)     # param_groups[-1]['lr']
        writer.add_scalar('training/va_loss', loss_va.val, curr_iter)
        writer.add_scalar('training/au_loss', loss_au.val, curr_iter)
        writer.add_scalar('training/expr_loss', loss_expr.val, curr_iter)
        writer.add_scalar('training/top1', top1.val, curr_iter)
        writer.add_scalar('training/au_strict', au_strict.val, curr_iter)
        # writer.add_scalar('training/au_soft', au_soft.val, curr_iter)


def validate(val_loader, model, criterion, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    loss_va = AverageMeter('VA CCC', ':.4f')
    loss_au = AverageMeter('AU Loss', ':.4f')
    loss_expr = AverageMeter('Expr Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    v_ccc = AverageMeter('V CCC', ':6.2f')
    a_ccc = AverageMeter('A CCC', ':6.2f')
    # top3 = AverageMeter('Acc@3', ':6.2f')
    au_strict = AverageMeter('AU Strict Acc', ':6.2f')
    # au_soft = AverageMeter('AU Soft Acc', ':6.2f')
    f1_expr = AverageMeter('Expr F1-score', ':6.2f')
    f1_aus = AverageMeter('AU F1-score', ':6.2f')
    # au_category = AverageMeter('AU Cgr Acc', '')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, loss_va, v_ccc, a_ccc, top1, f1_expr, au_strict, f1_aus],      # , au_category
        prefix="Validation Epoch: [{}]".format(epoch + 1)
    )
    # switch to evaluate mode, keeps BN features fixed
    curr_iter = epoch * len(val_loader)
    cm = 0
    mcm = 0
    model.eval()
    with torch.no_grad():
        end = time.time()
        for batch_index, (images, labels) in enumerate(val_loader):
            # bs, n_crops, channel, height, width = np.shape(images)
            # images = images.view(-1, channel, height, width)
            images = images.to(device)  # , torch.tensor(labels).to(device)
            v_labels = labels[:, 0].to(device)
            a_labels = labels[:, 1].to(device)
            # TODO: check AUs list, dont save AU labels separately, should be numpy array!!!
            aus_labels = labels[:, 2:-1].to(device)
            expr_labels = labels[:, -1].to(device)
            # target = (v_labels, a_labels, aus_labels, expr_labels)
            # compute output
            outputs = model(images)
            # if args.arch == "v2":
            va_output, aus_output, expr_output = outputs
                # v_output, a_output = va_output[:, 0], va_output[:, 1]
            # else:
            #     v_output, a_output, aus_output, expr_output = \
            #         outputs[:, 0], outputs[:, 1], outputs[:, 2:10], outputs[:, 10:]
            # compute loss
            if args.arcface:
                expr_output, expr_labels = metric_fc(expr_output, expr_labels)
                outputs = (va_output, aus_output, expr_output)

            loss, va_bs, au_bs, (va_loss, v_loss, a_loss, au_loss, expr_loss) = \
                criterion(outputs, v_labels, a_labels, aus_labels, expr_labels)

            # measure accuracy and record loss
            batch_size = images.size(0)
            # TODO: not check ignored index in acc of expr_labels
            (acc1, acc3), expr_bs, expr_cm, expr_prcn, expr_rcl, expr_f1 = \
                expr_accuracy(expr_output, expr_labels, topk=(1, 3), flag="val")
            # using total batch size as losses' batch size could result lower displayed loss.
            losses.update(loss.item(), batch_size)
            # va loss higher is better in val
            loss_va.update(1 - va_loss.item(), va_bs)
            loss_au.update(au_loss.item(), au_bs)
            loss_expr.update(expr_loss.item(), expr_bs)
            top1.update(acc1[0], expr_bs)
            v_ccc.update(v_loss, va_bs)
            a_ccc.update(a_loss, va_bs)
            # top3.update(acc3[0], expr_bs)
            f1_expr.update(expr_f1, expr_bs)
            strict_acc, au_mcm, au_prcn, au_rcl, au_f1, cgr_acc = \
                aus_accuracy(aus_output, aus_labels, threshold=0.5, flag="val")
            au_strict.update(strict_acc, au_bs)
            f1_aus.update(au_f1, au_bs)
            # au_category.update(cgr_acc, au_bs)
            # au_soft.update(soft_acc, au_bs)
            cm += expr_cm
            mcm += au_mcm
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            curr_iter += 1

            if batch_index % args.print_freq == 0 or batch_index == len(val_loader) - 1:
                info = progress.info(batch_index + 1)

                # msg = '[train main loss {:0.6f}], [VA loss {:0.6f}], [AUs loss {:0.6f}],' \
                #       '[Expr loss {:0.6f}], [AUs acc {:6.2f}, ], [Expr top1 {:6.2f}]'.format(
                #         losses.avg, loss_va.avg, loss_au.avg, loss_expr.avg, au_strict.avg, top1.avg)

                logging.info(info)

            writer.add_scalar('validating/loss', losses.val, curr_iter)  # (losses.val)
            writer.add_scalar('validating/va_ccc', loss_va.val, curr_iter)
            writer.add_scalar('validating/au_loss', loss_au.val, curr_iter)
            writer.add_scalar('validating/expr_loss', loss_expr.val, curr_iter)
            writer.add_scalar('validating/top1', top1.val, curr_iter)
            writer.add_scalar('validating/au_strict', au_strict.val, curr_iter)
            # writer.add_scalar('validating/au_soft', au_soft.val, curr_iter)
            if expr_prcn != -1 and expr_rcl != -1 and expr_f1 != -1:
                writer.add_scalar('validating/expr_prcn', expr_prcn, curr_iter)
                writer.add_scalar('validating/expr_rcl', expr_rcl, curr_iter)
                writer.add_scalar('validating/expr_f1', expr_f1, curr_iter)
            if au_prcn != -1 and au_rcl != -1 and au_f1 != -1:
                writer.add_scalar('validating/au_prcn', au_prcn, curr_iter)
                writer.add_scalar('validating/au_rcl', au_rcl, curr_iter)
                writer.add_scalar('validating/au_f1', au_f1, curr_iter)

    return losses, loss_va, top1, au_strict, cm, mcm, f1_expr, f1_aus


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count == 0: self.count = 1e-10
        # if isinstance(self.sum, np.ndarray):
        #     with np.errstate(divide='ignore', invalid="ignore"):
        #         self.avg = np.nan_to_num(self.sum / self.count)
        # else:
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def info(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return '\t'.join(entries)

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    optimizer.state_dict()["param_groups"][-1]['lr'] = lr


def expr_accuracy(output, target, topk=(1,), flag="train"):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        # batch_size = target.size(0)
        # remove ignore_index in target and output
        ignore_index = -1
        shape = output.shape
        mask = target != ignore_index
        target = target[mask]
        output = output[mask].reshape(-1, shape[1])
        assert target.size(0) == output.size(0)
        batch_size = output.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        top1 = pred[:, 0].cpu().numpy()
        target_np = target.view(-1).cpu().numpy()
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            if batch_size == 0: batch_size = 1e-10
            res.append(correct_k.mul_(100.0 / batch_size))

        if flag == "val":
            if top1.size == 0 and target_np.size == 0:
                cm = 0
                precision, recall, F1_score = -1, -1, -1
            else:
                cm = sm.confusion_matrix(target_np, top1, labels=range(7))
                precision, recall, F1_score = statistic(target_np, top1)
            return res, batch_size, cm, precision, recall, F1_score
        else:
            if top1.size == 0 and target_np.size == 0:
                return res, batch_size, 0
            F1_score = sm.f1_score(target_np, top1, average="macro", zero_division=1)
            return res, batch_size, F1_score


def aus_accuracy(output, target, threshold=0.5, flag="train"):
    with torch.no_grad():
        # batch_size = target.size(0)
        # remove ignore_index in target and output
        # TODO: debug output[
        ignore_index = -1
        shape = output.shape
        mask = target != ignore_index
        target = target[mask].reshape(-1, shape[1])
        output = output[mask].reshape(-1, shape[1])
        batch_size = output.size(0)
        if batch_size == 0: batch_size = 1e10
        mhot_output = torch.sigmoid(output) > threshold
        # synchronize the matrix type with target
        mhot_output = mhot_output.type_as(target)
        # print("sigmoid output:", mhot_output)
        # strict: must match in every class
        mhot_output_np = mhot_output.cpu().numpy()
        target_np = target.cpu().numpy()
        # tp = np.where(target_np == 1, mhot_output_np, np.zeros_like(mhot_output_np)).sum(0)
        # n = target_np.sum(0)
        tp = (mhot_output_np == target_np).sum(0)
        # with np.errstate(divide='ignore', invalid="ignore"):
        ctg_acc = np.nan_to_num(tp / batch_size)
        strict_correct = sum(list(map(lambda x, y: torch.equal(x, y), mhot_output, target))) / batch_size
        """
        # soft: logical_and divide logical_or = accuracy
        logical_and = np.sum(np.logical_and(mhot_output_np, target_np), axis=1)
        logical_or = np.sum(np.logical_or(mhot_output_np, target_np), axis=1)
        acc_and_or = np.nan_to_num(logical_and / logical_or)
        soft_correct = np.nan_to_num(acc_and_or.mean())
        """

        if flag == "val":
            if mhot_output_np.size == 0 and target_np.size == 0:
                mcm = 0
                precision, recall, F1_score = -1, -1, -1
            else:
                mcm = sm.multilabel_confusion_matrix(target_np, mhot_output_np)
                precision, recall, F1_score = statistic(target_np, mhot_output_np)
            return strict_correct * 100, mcm, precision, recall, F1_score, ctg_acc
        else:
            if mhot_output_np.size == 0 and target_np.size == 0:
                return strict_correct * 100, 0
            F1_score = sm.f1_score(target_np, mhot_output_np, average="macro", zero_division=1)
            return strict_correct * 100, F1_score


def statistic(target, predict):
    precision = sm.precision_score(target, predict, average="macro", zero_division=1)
    recall = sm.recall_score(target, predict, average="macro", zero_division=1)
    F1_score = sm.f1_score(target, predict, average="macro", zero_division=1)
    return precision, recall, F1_score


def fast_hist(pred, true, num_classes):
    mask = (true >= 0) & (true < num_classes)
    hist = np.bincount(num_classes * true[mask].astype(int) + pred[mask].astype(int),
                       minlength=num_classes ** 2).reshape(num_classes, num_classes)
    tp = np.diag(hist)
    fp = hist.sum(axis=1) - tp      # fp axis=1 or 0?
    fn = hist.sum(axis=0) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return hist, precision, recall, f1


def save_checkpoint(state, is_best, epoch):
    filename = "ckpt@"+str(epoch)+".pth.tar"
    torch.save(state, os.path.join(args.exp_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(args.exp_path, filename),
                        os.path.join(args.exp_path, 'model_best@' + str(epoch) + '.pth.tar'))


if __name__ == '__main__':
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # torch.multiprocessing.set_start_method('spawn')
    main()
