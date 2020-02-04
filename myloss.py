import torch
import torch.nn as nn
# import torch.nn.functional as F

device = torch.device('cuda')


class MultiTaskLoss(nn.Module):
    def __init__(self, loss_type, loss_uncertainties, gamma=0):
        super(MultiTaskLoss, self).__init__()
        self.loss_type = loss_type
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1)
        self.bce = nn.BCEWithLogitsLoss()
        self.loss_uncertainties = loss_uncertainties
        self.gamma = gamma

    @staticmethod
    def valid(pred, target, ignore_index):
        if ignore_index is None:
            return pred, target
        mask = (target != ignore_index)
        shape = mask.shape
        vp = pred[mask].reshape(-1, shape[1])
        vt = target[mask].reshape(-1, shape[1])
        return vp, vt

    @staticmethod
    def cvalid(pred, target, ignore_index):
        if ignore_index is None:
            return pred, target
        mask = (target != ignore_index)
        vp = pred[mask]
        vt = target[mask]
        return vp, vt

    def CCCloss(self, pred, target):
        ignore_index = -10
        pred = pred.to(device)
        # print("vpt: {0}, vtt: {1}".format(pred.type(), target.type()))
        pred, target = self.cvalid(pred, target, ignore_index)
        # print("vp: {0}, vt: {1}".format(pred, truth))
        # TODO: remove .type and .to(device)
        # pred = pred.type(torch.DoubleTensor)
        assert pred.size(0) == target.size(0)
        # truth = truth.type(torch.DoubleTensor)
        # print("pred size:", pred.size())
        # print("pred type is ", pred.type())
        # print("truth size:", truth.size())
        # print("truth type is ", truth.type())
        mean_cent_prod = ((pred - pred.mean()) * (target - target.mean())).mean()
        # 1 - is to minimize when training, to minimize loss
        loss = 2 * mean_cent_prod / (pred.var() + target.var() + (pred.mean() - target.mean()) ** 2)
        loss[torch.isnan(loss)] = 0
        return loss, target.size(0)

    def BCELossWithIgnore(self, pred, target):
        ignore_index = -1
        # non_pad_mask = label.ne(ingore_index), loss = loss.masked_select(non_pad_mask)
        pred, target = self.valid(pred, target, ignore_index)
        # truths = truths.type(torch.DoubleTensor)
        target = target.to(device)
        assert pred.size(0) == target.size(0)
        # print("AU pred:{0}, truth:{1}".format(preds, truths))
        # loss_msm = nn.MultiLabelSoftMarginLoss()
        loss = self.bce(pred, target)
        # loss_m = loss_msm(preds, truths)
        # torch.isnan(x) or x != x -> retval: mask mat where nan is 1
        loss[torch.isnan(loss)] = 0
        return loss, target.size(0)

    def CCEloss(self, pred, target):
        if pred.shape == torch.Size([0]) or target.shape == torch.Size([0]):
            return torch.tensor(0)
        return self.cross_entropy(pred, target.long())

    def Focalloss(self, pred, target):
        # print("pred size:", pred.size())
        # print("tagr size:", target.size())
        if pred.shape == torch.Size([0]) or target.shape == torch.Size([0]):
            return torch.tensor(0)
        else:
            logp = self.cross_entropy(pred, target.long())
            p = torch.exp(-logp)
            loss = (1 - p) ** self.gamma * logp
            return loss.mean()

    def calculate_total_loss(self, *losses):
        va_loss, au_loss, expr_loss = losses
        va_uncertainty, au_uncertainty, expr_uncertainty = self.loss_uncertainties

        loss = 0

        if self.loss_type == 'fixed':
            loss = va_loss + au_loss + expr_loss

        elif self.loss_type == 'learned':
            # va regression
            # 0.5 * (torch.exp(-va_uncertainty) * va_loss + va_uncertainty)
            loss += 0.5 * (torch.exp(-2 * va_uncertainty) * va_loss + va_uncertainty ** 2)

            # au classification
            # torch.exp(-au_uncertainty) * au_loss + 0.5 * au_uncertainty
            loss += 0.5 * (torch.exp(-2 * au_uncertainty) * au_loss + au_uncertainty ** 2)

            # expr classification
            # torch.exp(-expr_uncertainty) * expr_loss + 0.5 * expr_uncertainty
            loss += 0.5 * (torch.exp(-2 * expr_uncertainty) * expr_loss + expr_uncertainty ** 2)

        else:
            raise ValueError

        return loss

    def forward(self, pred, *target):
        va_pred, au_pred, expr_pred = pred
        v_pred, a_pred = va_pred[:, 0], va_pred[:, 1]
        v_target, a_target, au_target, expr_target = target

        v_loss, va_bs = self.CCCloss(v_pred, v_target)
        a_loss, _ = self.CCCloss(a_pred, a_target)
        au_loss, au_bs = self.BCELossWithIgnore(au_pred, au_target)
        # if args.arcface:
        #     expr_loss = self.Focalloss(expr_pred, expr_target)
        # else:
        expr_loss = self.CCEloss(expr_pred, expr_target)
        va_loss = 1 - 0.5 * (v_loss + a_loss)

        total_loss = self.calculate_total_loss(va_loss, au_loss, expr_loss)

        return total_loss, va_bs, au_bs, (va_loss, v_loss, a_loss, au_loss, expr_loss)
