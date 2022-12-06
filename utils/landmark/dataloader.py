import logging
from itertools import repeat
from multiprocessing.pool import ThreadPool

import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from utils.general import LOGGER, reconvert_np, colorstr
from utils.landmark.transforms import *
from utils.torch_utils import torch_distributed_zero_first

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


class WiderFaceDataset(Dataset):
    def __init__(self, txt_path, transform=None, is_train=True):
        self.transform = transform
        self.is_train = is_train

        if is_train:
            self.path = os.path.join(txt_path, 'WIDER_train')
        else:
            self.path = os.path.join(txt_path, 'WIDER_val')

        self.imgs_path = []
        self.words = []
        self.create_targets()

    def create_targets(self):
        txt_path = os.path.join(self.path, 'label.txt')
        f = open(txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt', 'images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 15))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 15))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]  # l0_x
            annotation[0, 5] = label[5]  # l0_y
            annotation[0, 6] = label[7]  # l1_x
            annotation[0, 7] = label[8]  # l1_y
            annotation[0, 8] = label[10]  # l2_x
            annotation[0, 9] = label[11]  # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y
            if annotation[0, 4] < 0:
                annotation[0, 14] = -1
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.transform is not None:
            img, target = self.transform(img, target)

        return torch.from_numpy(img), target


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return torch.stack(imgs, 0), targets


# ----------------------------
# yolo type dataloader

# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [x.replace(sa, sb, 1).replace('.' + x.split('.')[-1], '.txt') for x in img_paths]


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix=''):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadFaceImagesAndLabels(path, imgsz, batch_size,
                                          augment=augment,  # augment images
                                          hyp=hyp,  # augmentation hyperparameters
                                          rect=rect,  # rectangular training
                                          cache_images=cache,
                                          single_cls=opt.single_cls,
                                          stride=int(stride),
                                          pad=pad,
                                          image_weights=image_weights,
                                          prefix=prefix)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadFaceImagesAndLabels.collate_fn4 if quad else LoadFaceImagesAndLabels.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadFaceImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='', rgb_means=(104, 117, 123)):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.rgb_means = rgb_means

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                else:
                    raise FileNotFoundError(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\n{help_url}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)  # labels
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')  # cached labels
        if cache_path.is_file():
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files) or 'results' not in cache:  # changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        # Display cache
        [nf, nm, ne, nc, n] = cache.pop('results')  # found, missing, empty, corrupted, total
        desc = f"Scanning '{cache_path}' for images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        tqdm(None, desc=prefix + desc, total=n, initial=n)
        assert nf > 0 or not augment, f'{prefix}No labels found in {cache_path}, can not start training. {help_url}'

        # Read cache
        cache.pop('hash')  # remove hash
        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))  # 8 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x  # img, hw_original, hw_resized = load_image(self, i)
                gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'

    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # number missing, found, empty, duplicate
        desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels..."
        pbar = tqdm(zip(self.img_files, self.label_files), desc=desc, total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify()  # PIL verify
                shape = exif_size(im)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels
                    l = self.clap_out_of_bound(shape, l)
                    if not len(l):
                        ne += 1  # label empty
                        l = np.zeros((0, 15), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 15), dtype=np.float32)
                x[im_file] = [l, shape]
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label %s: %s' % (im_file, e))

            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' for images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"

        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = [nf, nm, ne, nc, i + 1]
        torch.save(x, path)  # save for next time
        logging.info(f"{prefix}New cache created: {path}")
        return x

    def clap_out_of_bound(self, shape, labels):
        # Prevention of cross the border
        cls = labels[:, :1] + 1
        bbox = labels[:, 1:5]
        # print("bbox",bbox)
        lmk = labels[:, 5:]
        # print("lmk",lmk)
        obbox = reconvert_np(shape, bbox)
        olmk = reconvert_lmk_np(shape, lmk)
        l1 = lmk[:, :1] < 0
        cls[l1] = -1

        slmk = torch.from_numpy(olmk)
        slmk = torch.clamp(slmk, max=shape[0]).numpy()
        sbbox = torch.from_numpy(obbox)
        sbbox = torch.clamp(sbbox, max=shape[0]).numpy()

        lmk_ = convert_lmk(shape, slmk)
        # bbox_ = convert_scale(shape,sbbox)
        bbox_ = convert(shape, sbbox)
        lmk_[lmk_[:, :] < 0] = -1
        # print("bbox_",bbox_)
        # print("lmk_",lmk_)
        # return np.hstack((cls, sbbox, slmk))
        return np.hstack((cls, bbox_, lmk_))

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):
        index = self.indices[index]
        image, (h0, w0), (h, w) = load_image(self, index)
        targets = self.labels[index]

        boxes = targets[:, 1:5].copy()
        labels = targets[:, 0].copy()
        landm = targets[:, 5:].copy()

        image_t, boxes_t, labels_t, landm_t, pad_image_flag = crop(image, boxes, labels, landm, self.img_size)
        image_t = distort(image_t)
        image_t = pad_to_square(image_t, self.rgb_means, pad_image_flag)
        image_t, boxes_t, landm_t = flip_horizontal(image_t, boxes_t, landm_t)
        height, width, _ = image_t.shape
        image_t = resize_norm(image_t, self.img_size, self.rgb_means)
        boxes_t[:, 0::2] /= width
        boxes_t[:, 1::2] /= height

        landm_t[:, 0::2] /= width
        landm_t[:, 1::2] /= height

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, landm_t, labels_t))

        return torch.from_numpy(image_t), targets_t, self.img_files[index], (h0, w0)

    @staticmethod
    def collate_fn(batch):
        targets = []
        imgs = []
        for _, sample in enumerate(batch):
            for _, tup in enumerate(sample):
                if torch.is_tensor(tup):
                    imgs.append(tup)
                elif isinstance(tup, type(np.empty(0))):
                    annos = torch.from_numpy(tup).float()
                    targets.append(annos)

        return torch.stack(imgs, 0), targets


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        oimg = img.copy()
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return oimg, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def load_data(train_path, val_path, imgsz, batch_size, opt, hyp, save_dir):
    if opt.dataset_type == 'voc':
        train_transform = Traintrans(imgsz[0], np.array(hyp['IMAGE_MEAN']))
        test_transform = Traintrans(imgsz[0], np.array(hyp['IMAGE_MEAN']))
        datasets = []
        for dataset_path in [train_path]:
            dataset = WiderFaceDataset(dataset_path, transform=train_transform, is_train=True)
            label_file = save_dir / "voc-model-labels.txt"
            datasets.append(dataset)
        LOGGER.info(colorstr('dataloader: ') + f"Stored labels into file {label_file}.")
        train_dataset = ConcatDataset(datasets)
        LOGGER.info("Train dataset size: {}".format(len(train_dataset)))
        train_loader = DataLoader(train_dataset, batch_size, num_workers=opt.workers, shuffle=True,
                                  collate_fn=detection_collate)
        # voc datasets train and valid data in the same file
        val_dataset = WiderFaceDataset(val_path, transform=test_transform, is_train=False)
        LOGGER.info("validation dataset size: {}".format(len(val_dataset)))
        val_loader = DataLoader(val_dataset, batch_size, num_workers=opt.workers, shuffle=False,
                                collate_fn=detection_collate)

    elif opt.dataset_type == 'yolo':
        # yolo type dataset still have some bugs
        train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, 32, opt, hyp=hyp,
                                                  augment=True, cache=None if opt.cache == 'val' else opt.cache,
                                                  rect=opt.rect, rank=LOCAL_RANK, workers=opt.workers,
                                                  image_weights=opt.image_weights, quad=opt.quad,
                                                  prefix=colorstr('train: '),
                                                  )

        if RANK in {-1, 0}:
            val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, 32, opt, hyp=hyp,
                                           cache=None if opt.noval else opt.cache, rect=True, rank=-1,
                                           workers=opt.workers * 2, pad=0.5, prefix=colorstr('val: '))[0]
    else:
        train_loader, val_loader, num_classes = None, None, None
        LOGGER.error(colorstr('dataloader: ') + f"Unsupported Dataset.")
        assert train_loader is not None

    return train_loader, val_loader


class LoadImages:
    def __init__(self, path, img_size=640, auto=True, transforms=None, vid_stride=1, oresize=False):
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms  # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        self.origin_size = oresize
        if any(videos):
            self._new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1
            # im0 = self._cv2_rotate(im0)  # for use if cv2 autorotation is False
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '

        else:
            # Read image
            self.count += 1
            im0 = cv2.imread(path)  # BGR
            assert im0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '

        if self.transforms:
            im = self.transforms(im0)  # transforms
            resize = 1
        else:
            max_size = self.img_size
            im_shape = im0.shape
            # ori_height,ori_width = im_shape[0],im_shape[1]
            im = np.float32(im0)
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            resize = float(self.img_size) / float(im_size_min)
            if np.round(resize * im_size_max) > max_size:
                resize = float(max_size) / float(im_size_max)
            if self.origin_size:
                resize = 1
            if resize != 1:
                im = cv2.resize(im, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            im_height, im_width, _ = im.shape

            # scale = torch.Tensor([im.shape[1], im.shape[0], im.shape[1], im.shape[0]])
            im -= (104, 117, 123)
            im = im.transpose(2, 0, 1)

        return path, im, im0, self.cap, s, resize

    def _new_video(self, path):
        # Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
        self.orientation = int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))  # rotation degrees
        # self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)  # disable https://github.com/ultralytics/yolov5/issues/8493

    def _cv2_rotate(self, im):
        # Rotate a cv2 video manually
        if self.orientation == 0:
            return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        elif self.orientation == 180:
            return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif self.orientation == 90:
            return cv2.rotate(im, cv2.ROTATE_180)
        return im

    def __len__(self):
        return self.nf  # number of files
