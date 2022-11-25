import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from model.net import create_net
from utils.box_utils import PriorBox, nms, convert_locations_to_boxes, center_form_to_corner_form
from utils.dataloader import LoadImages
from utils.general import colorstr, LOGGER, check_yaml, print_args, increment_path, check_img_size, Profile
from utils.transforms import PredictionTransform

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'face_lite.onnx', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'imgs', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[320, 240],
                        help='inference size w,h')
    parser.add_argument('--conf-thres', type=float, default=0.6, help='confidence threshold')
    parser.add_argument('--max-det', type=int, default=1500, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def postprocess(scores, boxes, priors, size, top_k=-1, prob_threshold=None, device=None,hyp=None):
    height, width, _ = size

    # from vision.ssd.config import fd_config as config
    scores = F.softmax(scores, dim=2)
    boxes = convert_locations_to_boxes(boxes, priors.to(device), hyp['CENTER_VAR'], hyp['SIZE_VAR'])
    boxes = center_form_to_corner_form(boxes)

    boxes = boxes[0]
    scores = scores[0]
    # this version of nms is slower on GPU, so we move data to CPU.
    boxes = boxes.to(device)
    scores = scores.to(device)
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, scores.size(1)):
        probs = scores[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.size(0) == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
        box_probs = nms(box_probs, nms_method=None,
                        score_threshold=prob_threshold,
                        iou_threshold=hyp['IOU_THRES'],
                        sigma=hyp['sigma'],
                        top_k=top_k,
                        candidate_size=hyp['candidate_size'])
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.size(0))
    if not picked_box_probs:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    picked_box_probs = torch.cat(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]


def detect(
        weights=ROOT / 'face_lite.pth',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        hyp=ROOT / 'data/hyps/hyp.scratch-low.yaml',
        imgsz=(320, 240),  # inference size (width, height)
        conf_thres=0.6,  # confidence threshold
        max_det=1500,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment

):
    source = str(source)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    hyp = check_yaml(hyp)
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    hyp = hyp.copy()

    # Load model
    weights = str(weights)
    onnx_model = weights.endswith('.onnx')
    if onnx_model:
        import onnxruntime
        net = onnxruntime.InferenceSession(weights)
        input_name = net.get_inputs()[0].name
    else:
        net = create_net()
        net.load(weights)
        net.eval()

    # Create priorbox
    priorbox = PriorBox(image_size=imgsz, hpy=hyp)
    with torch.no_grad():
        priors = priorbox.forward()

    imgsz = check_img_size(imgsz, hyp['IMAGE_SIZE_DICT'])  # verify imgsz is gs-multiple
    LOGGER.info(colorstr('image size: ') + f'{imgsz}')

    transform = PredictionTransform(imgsz, np.array(hyp['IMAGE_MEAN']), hyp['IMAGE_STD'])
    dataset = LoadImages(source, img_size=imgsz, transforms=transform)
    total = 0
    dt = (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            images = im.unsqueeze(0)

            if onnx_model:
                images = images.numpy()
                images = images.astype(np.float32) / 128.0
            else:
                images = images.to(device).float() / 128.0

        with dt[1]:
            with torch.no_grad():
                if onnx_model:
                    scores, boxes = net.run(None, {input_name: images})
                    scores, boxes = torch.as_tensor(scores), torch.as_tensor(boxes)
                else:
                    scores, boxes = net(images)
        with dt[2]:
            boxes, labels, probs = postprocess(scores, boxes, priors, im0.shape, top_k=max_det,
                                               prob_threshold=conf_thres, hyp=hyp)
        LOGGER.info(f"{s}{'' if boxes.size(0) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        total += boxes.size(0)
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            # lw = max(round(sum(im0.shape) / 2 * 0.003), 1)
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(im0, p1, p2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            # label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
            label = f"{probs[i]:.2f}"
            cv2.putText(im0, label, (int(box[0]), int(box[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(im0, str(boxes.size(0)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(save_path, im0)
        if view_img:
            cv2.imshow('result', im0)
            cv2.waitKey(0)
        LOGGER.info(f"Found {len(probs)} faces. The output image is {save_path}")
    LOGGER.info(f"Total face {total}")


if __name__ == "__main__":
    opt = parse_opt()
    detect(**vars(opt))
