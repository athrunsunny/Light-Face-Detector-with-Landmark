import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from model.landmark.net_nano import create_net
from utils.dataloader import LoadImages
from utils.general import colorstr, LOGGER, check_yaml, print_args, increment_path, Profile, scale_boxes, \
    scale_coords_landmarks
from utils.landmark.box_utils import PriorBox, decode_landm, decode, py_cpu_nms


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / '', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'imgs', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-lmk.yaml', help='hyperparameters path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=320,
                        help='inference size w,h')
    parser.add_argument('--conf-thres', type=float, default=0.02, help='confidence threshold')
    parser.add_argument('--max-det', type=int, default=1500, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--project', default=ROOT / 'runs/detect-lmk', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def postprocess(loc, conf, landms, size, resize, top_k=-1, keep_top_k=-1, nms_threshold=0.1, conf_thres=0.25,
                device=None, hyp=None):
    height, width = size
    scale = torch.Tensor([width, height, width, height])
    scale = scale.to(device)
    conf = F.softmax(conf, dim=2)
    priorbox = PriorBox(image_size=(height, width), hyp=hyp)
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data

    boxes = decode(loc.data.squeeze(0), prior_data, [hyp['CENTER_VAR'], hyp['SIZE_VAR']])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()

    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, [hyp['CENTER_VAR'], hyp['SIZE_VAR']])
    scale1 = torch.Tensor([width, height, width, height, width, height, width, height, width, height])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > conf_thres)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    return dets


def detect(
        weights=ROOT / '',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        hyp=ROOT / 'data/hyps/hyp.scratch-lmk.yaml',
        imgsz=320,  # inference size
        conf_thres=0.05,  # confidence threshold
        max_det=1500,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        origin_size=False,  # use origin image size to evaluate
        top_k=5000,
        nms_threshold=0.4,
        vis_thres=0.6,
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
        input_size = net.get_inputs()[0].shape
        onnx_input_size = input_size[2:]
        if imgsz != onnx_input_size:
            LOGGER.info(colorstr('size info: ') + f'Your input imgsz {imgsz},but onnx get input size {onnx_input_size},'
                                                  f'Now change input size to {onnx_input_size}')
            imgsz = onnx_input_size
    else:
        net = create_net()
        net.load(weights)
        net.eval()

    LOGGER.info(colorstr('image size: ') + f'{imgsz}')

    dataset = LoadImages(source, img_size=imgsz, oresize=origin_size, onnx=onnx_model, auto=False)
    total = 0
    dt = (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s, resize in dataset:
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # im.jpg
        with dt[0]:
            images = torch.from_numpy(im).to(device)
            if len(images.shape) == 3:
                images = images[None]
            if onnx_model:
                images = images.numpy()
                images = images.astype(np.float32)
            else:
                images = images.to(device).float()

        with dt[1]:
            with torch.no_grad():
                if onnx_model:
                    boxes, scores, landmarks = net.run(None, {input_name: images})
                    boxes, scores, landmarks = torch.as_tensor(boxes), torch.as_tensor(scores), torch.as_tensor(
                        landmarks)
                else:
                    boxes, scores, landmarks = net(images)
        with dt[2]:
            dets = postprocess(boxes, scores, landmarks, images.shape[2:], resize, top_k=top_k, keep_top_k=max_det // 2,
                               nms_threshold=nms_threshold, conf_thres=conf_thres, hyp=hyp)
            if onnx_model:
                dets = torch.as_tensor(dets)
                dets[:, :4] = scale_boxes(images.shape[2:], dets[:, :4], im0.shape).round()
                dets[:, 5:15] = scale_coords_landmarks(images.shape[2:], dets[:, 5:15], im0.shape).round()
                dets = dets.numpy()
        LOGGER.info(f"{s}{'' if dets.shape[0] else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        count_face = 0
        for b in dets:
            if b[4] < vis_thres:
                continue
            count_face += 1
            probs = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(im0s, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(im0s, probs, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

            # lmks
            cv2.circle(im0s, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(im0s, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(im0s, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(im0s, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(im0s, (b[13], b[14]), 1, (255, 0, 0), 4)

        cv2.imwrite(save_path, im0s)
        if view_img:
            cv2.imshow('result', im0s)
            cv2.waitKey(0)
        total += count_face
        LOGGER.info(f"Found {count_face} faces. The output image is {save_path}")
    LOGGER.info(f"Total face {total}")


if __name__ == "__main__":
    opt = parse_opt()
    detect(**vars(opt))
