import numpy as np
from .log import logger
from functools import partial
from mxnet.gluon.data.dataloader import default_mp_batchify_fn, default_batchify_fn


def save_checkpoint(net, checkpoints_path, epoch=None, prefix='', verbose=True):
    if epoch is None:
        checkpoint_name = 'last_checkpoint.params'
    else:
        checkpoint_name = f'{epoch:03d}.params'

    if prefix:
        checkpoint_name = f'{prefix}_{checkpoint_name}'

    if not checkpoints_path.exists():
        checkpoints_path.mkdir(parents=True)

    checkpoint_path = checkpoints_path / checkpoint_name
    if verbose:
        logger.info(f'Save checkpoint to {str(checkpoint_path)}')
    net.save_parameters(str(checkpoint_path))


def get_unique_labels(mask):
    return np.nonzero(np.bincount(mask.flatten() + 1))[0] - 1


def get_dict_batchify_fn(num_workers):
    base_batchify_fn = default_mp_batchify_fn if num_workers > 0 else default_batchify_fn

    return partial(dict_batchify_fn, base_batchify_fn=base_batchify_fn)


def dict_batchify_fn(data, base_batchify_fn):
    if isinstance(data[0], dict):
        ret = {k: [] for k in data[0].keys()}
        for x in data:
            for k, v in x.items():
                ret[k].append(v)
        return {k: base_batchify_fn(v) for k, v in ret.items()}
    else:
        return base_batchify_fn(data)


def get_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def expand_bbox(bbox, expand_ratio, min_crop_size=None):
    rmin, rmax, cmin, cmax = bbox
    rcenter = 0.5 * (rmin + rmax)
    ccenter = 0.5 * (cmin + cmax)
    height = expand_ratio * (rmax - rmin + 1)
    width = expand_ratio * (cmax - cmin + 1)
    if min_crop_size is not None:
        height = max(height, min_crop_size)
        width = max(width, min_crop_size)

    rmin = int(round(rcenter - 0.5 * height))
    rmax = int(round(rcenter + 0.5 * height))
    cmin = int(round(ccenter - 0.5 * width))
    cmax = int(round(ccenter + 0.5 * width))

    return rmin, rmax, cmin, cmax


def clamp_bbox(bbox, rmin, rmax, cmin, cmax):
    return (max(rmin, bbox[0]), min(rmax, bbox[1]),
            max(cmin, bbox[2]), min(cmax, bbox[3]))


def get_bbox_iou(b1, b2):
    h_iou = get_segments_iou(b1[:2], b2[:2])
    w_iou = get_segments_iou(b1[2:4], b2[2:4])
    return h_iou * w_iou


def get_segments_iou(s1, s2):
    a, b = s1
    c, d = s2
    intersection = max(0, min(b, d) - max(a, c) + 1)
    union = max(1e-6, max(b, d) - min(a, c) + 1)
    return intersection / union
