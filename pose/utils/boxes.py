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


def xywh2xyxy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        boxes = x.clone()
    elif isinstance(x, np.ndarray):
        boxes = x.copy()
    else:
        raise TypeError("Input must be a tensor or numpy array")

    boxes[:, 0] = x[:, 0] - x[:, 2] / 2
    boxes[:, 1] = x[:, 1] - x[:, 3] / 2
    boxes[:, 2] = x[:, 0] + x[:, 2] / 2
    boxes[:, 3] = x[:, 1] + x[:, 3] / 2

    return boxes


def xyxy2xywh(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        y = x.clone()
    elif isinstance(x, np.ndarray):
        y = x.copy()
    else:
        raise TypeError("Input must be a tensor or numpy array")

    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height

    return y


def non_max_suppression(
    pred: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: list = None,
    max_det: int = 300,
) -> list:
    """
    Non-Maximum Suppression (NMS) on inference results

    Args:
        pred: predictions tensor (n,7) [x, y, w, h, obj_conf, cls1_conf, cls2_conf]
        conf_thres: confidence threshold
        iou_thres: NMS IoU threshold
        classes: filter by class (e.g. [0] for persons only)
        max_det: maximum number of detections per image

    Returns:
        list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # Ensure pred is 2D
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)

    # Calculate confidence
    conf = pred[:, 4]  # objectness score
    class_scores = pred[:, 5:]  # class probabilities
    class_conf, class_pred = class_scores.max(1)  # best class confidence and prediction
    confidence = conf * class_conf  # combine scores

    # Filter by confidence
    conf_mask = confidence > conf_thres
    pred = pred[conf_mask]
    confidence = confidence[conf_mask]
    class_pred = class_pred[conf_mask]

    if not pred.shape[0]:  # no boxes
        return [torch.zeros((0, 6), device=pred.device)]

    # Convert boxes from [x, y, w, h] to [x1, y1, x2, y2]
    boxes = xywh2xyxy(pred[:, :4])

    # Filter by class
    if classes is not None:
        if isinstance(classes, int):
            classes = [classes]
        class_mask = torch.zeros_like(class_pred, dtype=torch.bool)
        for c in classes:
            class_mask |= class_pred == c
        boxes = boxes[class_mask]
        confidence = confidence[class_mask]
        class_pred = class_pred[class_mask]

    if not boxes.shape[0]:  # no boxes after filtering
        return [torch.zeros((0, 6), device=pred.device)]

    # Sort by confidence
    sorted_indices = torch.argsort(confidence, descending=True)
    boxes = boxes[sorted_indices]
    confidence = confidence[sorted_indices]
    class_pred = class_pred[sorted_indices]

    # Apply NMS
    keep = ops.nms(boxes, confidence, iou_thres)
    if keep.shape[0] > max_det:
        keep = keep[:max_det]

    # Combine detections into final format [x1, y1, x2, y2, conf, cls]
    output = torch.zeros((keep.shape[0], 6), device=pred.device)
    output[:, :4] = boxes[keep]
    output[:, 4] = confidence[keep]
    output[:, 5] = class_pred[keep].float()

    return [output]
