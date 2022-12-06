import argparse
import os
import platform
import sys
from pathlib import Path

import onnx
import onnxsim
import pandas as pd
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import LOGGER, colorstr, file_size, print_args
from utils.torch_utils import select_device
from model.landmark.net_nano import create_net


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / '', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[240, 320], help='image (h,w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--include', nargs='+', default=['onnx'], help='Export mode')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--simplify', default=True, action='store_true', help='ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX: opset version')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def export_onnx(model, im, file, opset, train, simplify=True, prefix=colorstr('ONNX:')):
    # ONNX export
    try:

        LOGGER.info(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        f = file.with_suffix('.onnx')

        torch.onnx.export(
            model,
            im,
            f,
            verbose=False,
            opset_version=opset,
            training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
            do_constant_folding=not train,
            input_names=['images'],
            output_names=['boxes', 'scores','landmarks'])

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        onnx.save(model_onnx, f)

        # Simplify
        if simplify:
            try:
                LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                LOGGER.info(f'{prefix} simplifier failure: {e}')
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        return f
    except Exception as e:
        LOGGER.info(f'{prefix} export failure: {e}')


def export_formats():
    #  export formats
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['ONNX', 'onnx', '.onnx', True, True], ]
    return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


def run(
        weights=ROOT / 'face_lite.pth',  # weights path
        imgsz=(240, 320),  # image (height, width)
        batch_size=1,  # batch size
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        include=('onnx'),  # include formats
        train=False,  # model.train() mode
        simplify=True,  # ONNX: simplify model
        opset=12,  # ONNX: opset version

):
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()['Argument'][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'

    file = Path(weights)  # PyTorch weights
    # Load PyTorch model
    device = select_device(device)
    net = create_net()
    net.load(weights)

    net.eval()
    im = torch.zeros(batch_size, 3, *imgsz).to(device)
    export_onnx(net, im, file, opset, train, simplify)


def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [opt.weights]):
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
