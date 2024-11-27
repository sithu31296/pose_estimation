import cv2
import numpy as np
import torch
from torchvision import ops


def letterbox(img, new_shape=(640, 640)):
    H, W = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / H, new_shape[1] / W)
    nH, nW = round(H * r), round(W * r)
    pH, pW = np.mod(new_shape[0] - nH, 32) / 2, np.mod(new_shape[1] - nW, 32) / 2

    if (H, W) != (nH, nW):
        img = cv2.resize(img, (nW, nH), interpolation=cv2.INTER_LINEAR)

    top, bottom = round(pH - 0.1), round(pH + 0.1)
    left, right = round(pW - 0.1), round(pW + 0.1)
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return img


def scale_boxes(boxes, orig_shape, new_shape):
    H, W = orig_shape
    nH, nW = new_shape
    gain = min(nH / H, nW / W)
    pad = (nH - H * gain) / 2, (nW - W * gain) / 2

    boxes[:, ::2] -= pad[1]
    boxes[:, 1::2] -= pad[0]
    boxes[:, :4] /= gain

    boxes[:, ::2].clamp_(0, orig_shape[1])
    boxes[:, 1::2].clamp_(0, orig_shape[0])
    return boxes.round()


def xywh2xyxy(x):
    boxes = x.clone()
    boxes[:, 0] = x[:, 0] - x[:, 2] / 2
    boxes[:, 1] = x[:, 1] - x[:, 3] / 2
    boxes[:, 2] = x[:, 0] + x[:, 2] / 2
    boxes[:, 3] = x[:, 1] + x[:, 3] / 2
    return boxes


def xyxy2xywh(x):
    y = x.clone()
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None):
    candidates = pred[..., 4] > conf_thres

    max_wh = 4096
    max_nms = 30000
    max_det = 300

    output = [torch.zeros((0, 6), device=pred.device)] * pred.shape[0]

    for xi, x in enumerate(pred):
        x = x[candidates[xi]]

        if not x.shape[0]:
            continue

        # compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # box
        box = xywh2xyxy(x[:, :4])

        # detection matrix nx6
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat([box, conf, j.float()], dim=1)[conf.view(-1) > conf_thres]

        # filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # check shape
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # batched nms
        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]
        keep = ops.nms(boxes, scores, iou_thres)

        if keep.shape[0] > max_det:
            keep = keep[:max_det]

        output[xi] = x[keep]

    return output
