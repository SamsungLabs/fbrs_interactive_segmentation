import os
import argparse
import tkinter as tk

import mxnet as mx

from isegm.utils import exp
from isegm.inference import utils
from interactive_demo.app import InteractiveDemoApp
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


def main():
    args, cfg = parse_args()

    checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    model = utils.load_deeplab_is_model(checkpoint_path, args.ctx, num_max_clicks=args.n_clicks)

    root = tk.Tk()
    root.minsize(960, 480)
    app = InteractiveDemoApp(root, args, model)
    root.deiconify()
    app.mainloop()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='The path to the checkpoint. '
                             'This can be a relative path (relative to cfg.INTERACTIVE_MODELS_PATH) '
                             'or an absolute path. The file extension can be omitted.')

    parser.add_argument('--n-clicks', type=int, default=100,
                        help='Maximum number of input clicks for the model.')

    parser.add_argument('--gpu', type=int, default=0,
                        help='Id of GPU to use.')

    parser.add_argument('--cfg', type=str, default="config.yml",
                        help='The path to the config file.')

    args = parser.parse_args()
    args.ctx = mx.gpu(args.gpu)
    cfg = exp.load_config_file(args.cfg, return_edict=True)

    return args, cfg


if __name__ == '__main__':
    main()
