from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import transforms as T
from tqdm import tqdm

from pose.models import get_pose_model
from pose.utils.boxes import letterbox, non_max_suppression, scale_boxes, xyxy2xywh
from pose.utils.decode import get_final_preds, get_simdr_final_preds
from pose.utils.utils import (
    FPS,
    VideoReader,
    VideoWriter,
    WebcamStream,
    draw_bbox,
    draw_keypoints,
    get_affine_transform,
    setup_cudnn,
)


class Pose:
    def __init__(
        self,
        det_model: str,
        pose_model: str,
        img_size: int = 640,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> None:
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "yolov5" in det_model:
            self.det_model_type = "yolov5"
            self.det_model = torch.hub.load(
                "ultralytics/yolov5", "custom", path=det_model, force_reload=True
            )
            self.det_model = self.det_model.to(self.device)
        else:
            self.det_model_type = "yolo"
            self.det_model = YOLO(det_model)
            self.det_model = self.det_model.to(self.device)

        self.model_name = pose_model
        self.pose_model = get_pose_model(pose_model)
        self.pose_model.load_state_dict(torch.load(pose_model, map_location="cpu"))
        self.pose_model = self.pose_model.to(self.device)
        self.pose_model.eval()

        self.patch_size = (192, 256)

        self.pose_transform = T.Compose(
            [T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )

        self.coco_skeletons = [
            [16, 14],
            [14, 12],
            [17, 15],
            [15, 13],
            [12, 13],
            [6, 12],
            [7, 13],
            [6, 7],
            [6, 8],
            [7, 9],
            [8, 10],
            [9, 11],
            [2, 3],
            [1, 2],
            [1, 3],
            [2, 4],
            [3, 5],
            [4, 6],
            [5, 7],
        ]

    def preprocess(self, image):
        img = letterbox(image, new_shape=self.img_size)
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        img = img[None]
        return img

    def box_to_center_scale(self, boxes, pixel_std=200):
        boxes = xyxy2xywh(boxes)
        r = self.patch_size[0] / self.patch_size[1]
        mask = boxes[:, 2] > boxes[:, 3] * r
        boxes[mask, 3] = boxes[mask, 2] / r
        boxes[~mask, 2] = boxes[~mask, 3] * r
        boxes[:, 2:] /= pixel_std
        boxes[:, 2:] *= 1.25
        return boxes

    def predict_poses(self, boxes, img):
        image_patches = []
        for cx, cy, w, h in boxes:
            trans = get_affine_transform(
                np.array([cx, cy]), np.array([w, h]), self.patch_size
            )
            img_patch = cv2.warpAffine(
                img, trans, self.patch_size, flags=cv2.INTER_LINEAR
            )
            img_patch = self.pose_transform(img_patch)
            image_patches.append(img_patch)

        image_patches = torch.stack(image_patches).to(self.device)
        return self.pose_model(image_patches)

    def postprocess(self, pred, img1, img0):
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=0)

        for det in pred:
            if len(det):
                boxes = scale_boxes(det[:, :4], img0.shape[:2], img1.shape[-2:]).cpu()
                boxes = self.box_to_center_scale(boxes)
                outputs = self.predict_poses(boxes, img0)

                if "simdr" in self.model_name.lower():
                    coords = get_simdr_final_preds(*outputs, boxes, self.patch_size)
                else:
                    coords = get_final_preds(outputs, boxes)

                img0 = draw_keypoints(img0, coords, self.coco_skeletons)
                img0 = draw_bbox(img0, det.cpu().numpy())

    @torch.inference_mode()
    def predict(self, image):
        img = self.preprocess(image)
        pred = self.det_model(img)[0]
        self.postprocess(pred, img, image)
        return image


def argument_parser():
    parser = ArgumentParser(
        description="Pose Estimation",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="assests/test.jpg",
        help="Path to image, video or webcam",
    )
    parser.add_argument(
        "--det-model",
        type=str,
        default="checkpoints/crowdhuman_yolov5m.pt",
        help="Human detection model",
    )
    parser.add_argument(
        "--pose-model",
        type=str,
        default="checkpoints/pretrained/simdr_hrnet_w32_256x192.pth",
        help="Pose estimation model",
    )
    parser.add_argument("--img-size", type=int, default=640, help="Image size")
    parser.add_argument(
        "--conf-thres", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument("--iou-thres", type=float, default=0.5, help="IOU threshold")
    return parser.parse_args()


def main():
    setup_cudnn()
    args = argument_parser()
    pose = Pose(
        det_model=args.det_model,
        pose_model=args.pose_model,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
    )

    source = Path(args.source)

    if source.is_file() and source.suffix in [".jpg", ".png"]:
        image = cv2.imread(str(source))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output = pose.predict(image)
        cv2.imwrite(
            f"{str(source).rsplit('.', maxsplit=1)[0]}_out.jpg",
            cv2.cvtColor(output, cv2.COLOR_RGB2BGR),
        )

    elif source.is_dir():
        files = source.glob("*.jpg")
        for file in files:
            image = cv2.imread(str(file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output = pose.predict(image)
            cv2.imwrite(
                f"{str(file).rsplit('.', maxsplit=1)[0]}_out.jpg",
                cv2.cvtColor(output, cv2.COLOR_RGB2BGR),
            )

    elif source.is_file() and source.suffix in [".mp4", ".avi"]:
        reader = VideoReader(args.source)
        writer = VideoWriter(
            f"{args.source.rsplit('.', maxsplit=1)[0]}_out.mp4", reader.fps
        )
        fps = FPS(len(reader.frames))

        for frame in tqdm(reader):
            fps.start()
            output = pose.predict(frame.numpy())
            fps.stop(False)
            writer.update(output)

        print(f"FPS: {fps.fps}")
        writer.write()

    else:
        webcam = WebcamStream()
        fps = FPS()

        for frame in webcam:
            fps.start()
            output = pose.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fps.stop()
            cv2.imwrite("frame.jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
