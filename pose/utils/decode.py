import math

import numpy as np
import torch
from torch import Tensor


def get_simdr_final_preds(
    pred_x: Tensor, pred_y: Tensor, boxes: Tensor, image_size: tuple
):
    center, scale = boxes[:, :2].numpy(), boxes[:, 2:].numpy()

    pred_x, pred_y = pred_x.softmax(dim=2), pred_y.softmax(dim=2)
    pred_x, pred_y = pred_x.max(dim=2)[-1], pred_y.max(dim=2)[-1]
    coords = torch.stack([pred_x / 2, pred_y / 2], dim=-1).cpu().numpy()

    for i in range(coords.shape[0]):
        coords[i] = transform_preds(coords[i], center[i], scale[i], image_size)
    return coords.astype(int)


def get_final_preds(heatmaps: Tensor, boxes: Tensor):
    center, scale = boxes[:, :2].numpy(), boxes[:, 2:].numpy()
    heatmaps = heatmaps.cpu().numpy()
    B, C, H, W = heatmaps.shape
    coords = get_max_preds(heatmaps)

    for n in range(B):
        for p in range(C):
            hm = heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))

            if 1 < px < W - 1 and 1 < py < H - 1:
                diff = np.array(
                    [hm[py][px + 1] - hm[py][px - 1], hm[py + 1][px] - hm[py - 1][px]]
                )
                coords[n][p] += np.sign(diff) * 0.25

    for i in range(B):
        coords[i] = transform_preds(coords[i], center[i], scale[i], [W, H])
    return coords.astype(int)


def get_max_preds(heatmaps: np.ndarray):
    B, C, _, W = heatmaps.shape
    heatmaps = heatmaps.reshape((B, C, -1))
    idx = np.argmax(heatmaps, axis=2).reshape((B, C, 1))
    maxvals = np.amax(heatmaps, axis=2).reshape((B, C, 1))
    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W
    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds


def transform_preds(coords, center, scale, output_size):
    scale = scale * 200
    scale_x = scale[0] / output_size[0]
    scale_y = scale[1] / output_size[1]
    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
    return target_coords
