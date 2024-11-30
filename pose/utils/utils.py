import cv2
import numpy as np
import torch
import time
from torchvision import io
from threading import Thread
from torch.backends import cudnn


def setup_cudnn() -> None:
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    cudnn.benchmark = True
    cudnn.deterministic = False


def draw_bbox(
    img: np.ndarray,
    boxes: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.5,
    text_color: tuple[int, int, int] = (255, 255, 255),
    text_thickness: int = 1,
) -> np.ndarray:
    for box in boxes:
        x1, y1, x2, y2, conf, _ = map(int, box[:6])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        conf_text = f"{conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            conf_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        cv2.rectangle(
            img,
            (x1, y1 - text_height - 5),
            (x1 + text_width + 5, y1),
            color,
            -1,  # Filled rectangle
        )
        cv2.putText(
            img,
            conf_text,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            text_thickness,
        )

    return img


def draw_coco_keypoints(img, keypoints, skeletons):
    if keypoints == []:
        return img
    image = img.copy()
    for kpts in keypoints:
        for x, y, v in kpts:
            if v == 2:
                cv2.circle(image, (x, y), 4, (255, 0, 0), 2)
        for kid1, kid2 in skeletons:
            x1, y1, v1 = kpts[kid1 - 1]
            x2, y2, v2 = kpts[kid2 - 1]
            if v1 == 2 and v2 == 2:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def draw_keypoints(img, keypoints, skeletons):
    if len(keypoints) == 0 or (
        isinstance(keypoints, np.ndarray) and keypoints.size == 0
    ):
        return img

    for kpts in keypoints:
        for x, y in kpts:
            cv2.circle(img, (x, y), 4, (255, 0, 0), 2, cv2.LINE_AA)
        for kid1, kid2 in skeletons:
            cv2.line(img, kpts[kid1 - 1], kpts[kid2 - 1], (0, 255, 0), 2, cv2.LINE_AA)

    return img


class WebcamStream:
    def __init__(self, src=0) -> None:
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        assert self.cap.isOpened(), f"Failed to open webcam {src}"
        _, self.frame = self.cap.read()
        Thread(target=self.update, args=([]), daemon=True).start()

    def update(self):
        while self.cap.isOpened():
            _, self.frame = self.cap.read()

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        if cv2.waitKey(1) == ord("q"):
            self.stop()

        return self.frame.copy()

    def stop(self):
        cv2.destroyAllWindows()
        raise StopIteration

    def __len__(self):
        return 0


class VideoReader:
    def __init__(self, video: str):
        self.frames, _, info = io.read_video(video, pts_unit="sec")
        self.fps = info["video_fps"]

        print(f"Processing '{video}'...")
        print(f"Total Frames: {len(self.frames)}")
        print(f"Video Size  : {list(self.frames.shape[1:-1])}")
        print(f"Video FPS   : {self.fps}")

    def __iter__(self):
        self.count = 0
        return self

    def __len__(self):
        return len(self.frames)

    def __next__(self):
        if self.count == len(self.frames):
            raise StopIteration
        frame = self.frames[self.count]
        self.count += 1
        return frame


class VideoWriter:
    def __init__(self, file_name, fps):
        self.fname = file_name
        self.fps = fps
        self.frames = []

    def update(self, frame):
        if isinstance(frame, np.ndarray):
            frame = torch.from_numpy(frame)
        self.frames.append(frame)

    def write(self):
        print(f"Saving video to '{self.fname}'...")
        io.write_video(self.fname, torch.stack(self.frames), self.fps)


class FPS:
    def __init__(self, avg=10) -> None:
        self.accum_time = 0
        self.counts = 0
        self.avg = avg

    def synchronize(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self):
        self.synchronize()
        self.prev_time = time.time()

    def stop(self, debug=True):
        self.synchronize()
        self.accum_time += time.time() - self.prev_time
        self.counts += 1
        if self.counts == self.avg:
            self.fps = round(self.counts / self.accum_time)
            if debug:
                print(f"FPS: {self.fps}")
            self.counts = 0
            self.accum_time = 0


def get_dir(src_point, rot):
    rot_rad = np.pi * rot / 180
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    p1 = src_point[0] * cs - src_point[1] * sn
    p2 = src_point[0] * sn + src_point[1] * cs
    return p1, p2


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_affine_transform(center, scale, patch_size, rot=0, inv=False):
    shift = np.array([0, 0], dtype=np.float32)
    scale_tmp = scale * 200
    src_w = scale_tmp[0]
    dst_w = patch_size[0]
    dst_h = patch_size[1]

    src_dir = get_dir([0, src_w * -0.5], rot)
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    return cv2.getAffineTransform(dst, src) if inv else cv2.getAffineTransform(src, dst)
