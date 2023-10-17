# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets, it's 0

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:  # g is zero
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7, balance is [4,1,0.4]
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index . ssi = 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance  # autobalance = false.
        self.na = m.na  # number of anchors = 3
        self.nc = m.nc  # number of classes = 80
        self.nl = m.nl  # number of layers = 3, this is number of output layers.
        self.anchors = m.anchors  # predefined anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        '''
        # shape of pred is list 1)  16x3x80x80x85 2) 16x3x40x40x85 3) 16x3x20x20x85
        # target is 197x6， targets[:,0] is image rank in batch. so that the algorithm can know which image the target belongs to.
        '''
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image_rk, anchor_rk, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # shape is 16,3,80,80 ,values are all zero  # target obj # it stores only the iou ratio.

            n = b.shape[0]  # number of targets 937, including repeated ones.
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5 # xx.sigmoid() will apply sigmoid to each element in the tensor.
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i] # (2*sigmoid(x))^2 * anchor.
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target) # pbox = (937,4) tbox[i] = (937,4) # shape is (937,)  # the code calculates CIoU
                lbox += (1.0 - iou).mean()  # iou loss . lbox : the small the better . iou : the larger the better.

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou: # do not sort iou
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1: # do not perform gradient reversal
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets , pcls shape is 1173,80
                    t[range(n), tcls[i]] = self.cp # class prob = 1 . t shape is 1173,80
                    lcls += self.BCEcls(pcls, t)  # BCE loss for class.

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)  # the pi's shape is 16x3x80x80x85, and the pi[...,4] gets 16x3x80x80.
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance: # ignored
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']  # box loss weight is 0.05
        lobj *= self.hyp['obj']  # obj weight is 1
        lcls *= self.hyp['cls']  # cls weight is 0.5
        bs = tobj.shape[0]  # batch size = 16

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        '''
        the function is used to process the raw targets. making it more suitable for the loss function.
        detail of the function:
        1.it will delete targets that it's box is different from the anchors.
        2.it introduces offset to the targets.

        return values:
        indices : [img_rank, anchor_rank, grid_y, grid_x]
        tbox : [offset_x,offset_y,raito_w,ratio_h] . the offset is relative to the left-top corner of the grid.
        anch : [anchor_w, anchor_h]
        tcls: [class]
        '''
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h). original target shape is [275,6]
        na, nt = self.na, targets.shape[0]  # number of anchors, nt = number of boxes in a batch
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1,
                                                                             nt)  # shape is [predefined_anchor, boxes_num]      # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]),
                            2)  # append anchor indices [3,275,7], add anchor rank to the last column.

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],  # x move right 1 grid
                [0, 1],  # y move down 1 grid
                [-1, 0],  # x move left 1 grid
                [0, -1],  # j,k,l,m # y move up 1 grid
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets
        # off shape is [5,2]
        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[
                i].shape  # get the 1st element from list. example shape is [16,3,80,80,85]
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain. 3232 is index.

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7) # shape is 3,275,7 : recover xywh from ratio to absolute point.
            if nt:  # nt is number of target
                # Matches
                r = t[..., 4:6] / anchors[:,
                                  None]  # wh ratio, [3,275,2] / [3,2] = [3,275,2]. example : [0][x][1]/[0][1]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare， anchor_t is 4. shape is [3,275]
                #
                t = t[j]  # filter. and it's shape is [all true obj, 7 ]

                # Offsets gxy means grid. gxi means means grid inverse.
                gxy = t[:, # shape is [obj_num, 2]
                      2:4]  # grid xy(center: offset relative to the left-top corner). get xy from ratio<4 labels. offset from top-left
                gxi = gain[[2,
                            3]] - gxy  # inverse (flip) grid xy  :  real - real * [x,y,w,h] . offset from right-bottom.
                j, k = ((gxy % 1 < g) & (gxy > 1)).T  # g = 0.5， find the grid xy that the remainder is less than 0.5
                l, m = ((gxi % 1 < g) & (gxi > 1)).T  # g = 0.5， find the grid xy that the remainder is less than 0.5. it selects the rows.
                j = torch.stack((torch.ones_like(j), j, k, l,
                                 m))  # 5,391. [all_true, x_true, y_true, x_inverse_true, y_inverse_true]
                t = t.repeat((5, 1, 1))[j]  # 391,7 -> repeat 5 times -> 1955,7 -> select true ones -> 1173,7
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] # [1,314,2] + [5,1,2] = [5,314,2]. and apply j to get => [1173,2]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors rank, image, class
            gij = (gxy - offsets).long() # long makes float becomes integer
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1),
                            gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid => (i,j) or (row_num, col_num)
            tbox.append(torch.cat((gxy - gij, gwh),
                                  1))  # box . gxy is the float point. gij is the integer point. so it stores the offset to the grid point.
            anch.append(anchors[a])  # anchors . from index to real anchor value.
            tcls.append(c)  # class. => img's label

        return tcls, tbox, indices, anch
