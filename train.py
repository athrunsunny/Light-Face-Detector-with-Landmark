import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

from model.net import create_net
from utils.box_utils import PriorBox
from utils.dataloader import load_data
from utils.general import init_seeds, print_args, check_file, get_latest_run, check_yaml, increment_path, LOGGER, \
    check_img_size, colorstr, one_cycle, check_suffix, check_dataset
from utils.loss import MultiboxLoss
from utils.torch_utils import select_device, smart_optimizer

warnings.filterwarnings("ignore", category=Warning)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'face_lite.pth', help='initial weights path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/widerface.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=320, help='train, val image size (pixels)')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')
    # Train params
    parser.add_argument('--val_epochs', default=5, type=int, help='the number epochs')
    parser.add_argument("--dataset_type", default="voc", type=str, choices=['voc', 'yolo'], help='Specify dataset type')
    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))

    # Resume (from specified or most recent last.pt)
    if opt.resume and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # replace
        opt.weights, opt.resume = str(last), True  # reinstate
        opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device)


def train(hyp, opt, device):
    save_dir, epochs, batch_size, weights, evolve, data, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.evolve, opt.data, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    cuda = device.type != 'cpu'
    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()

    # Image size
    imgsz = check_img_size(opt.imgsz, hyp['IMAGE_SIZE_DICT'])  # verify imgsz is gs-multiple
    LOGGER.info(colorstr('image size: ') + f'{imgsz}')

    # Model
    check_suffix(weights, '.pth')  # check weights
    pretrained = weights.endswith('.pth')
    net = create_net()
    if pretrained:
        state_dict = torch.load(weights, map_location='cpu')
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
        LOGGER.info(f'Transferred {len(net.state_dict())} items from {weights}')  # report

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING: DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        net = torch.nn.DataParallel(net, device_ids=device)

    # Create priorbox
    priorbox = PriorBox(image_size=(imgsz[0], imgsz[1]), hpy=hyp)
    with torch.no_grad():
        priors = priorbox.forward()

    # Dataloader
    data_dict = None
    data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    train_loader, val_loader = load_data(train_path, val_path, imgsz, priors, batch_size, opt, hyp, save_dir)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(net, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    net.to(device)
    criterion = MultiboxLoss(priors, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2, device=device)

    scheduler.last_epoch, last_epoch = -1, -1
    LOGGER.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, opt.epochs):
        pbar = enumerate(train_loader)
        nb = len(train_loader)
        LOGGER.info(
            ('\n' + '%11s' * 8) % ('Epoch', 'GPU_mem', 'avg_loss', 'box_loss', 'cls_loss', 'Instances', 'LR', 'Size'))
        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        net.train(True)
        running_loss = 0.0
        running_regression_loss = 0.0
        running_classification_loss = 0.0
        num = 0
        for i, data in pbar:  # batch -------------------------------------------------------------
            images, boxes, labels = data
            images = images.to(device).float() / 128.0
            boxes = boxes.to(device)
            labels = labels.to(device)
            boxes = boxes.to(torch.float32)

            optimizer.zero_grad()
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_regression_loss += regression_loss.item()
            running_classification_loss += classification_loss.item()
            num += images.shape[0]

            avg_loss = running_loss / num
            avg_reg_loss = running_regression_loss / num
            avg_clf_loss = running_classification_loss / num

            lr = optimizer.param_groups[0]['lr']
            if RANK in {-1, 0}:
                mloss = (avg_loss, avg_reg_loss, avg_clf_loss)
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 6) %
                                     (f'{epoch}/{opt.epochs - 1}', mem, *mloss, labels.shape[0], lr, images.shape[-1]))
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        scheduler.step()

        if epoch % opt.val_epochs == 0 or epoch == opt.epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, device)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(w, f"slim-Epoch-{epoch}-Loss-{val_loss}.pth")
            torch.save(net.state_dict(), model_path)
            LOGGER.info(f"Saved model {model_path}")


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device).float() / 128.0
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += images.shape[0]
        boxes = boxes.to(torch.float32)

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
