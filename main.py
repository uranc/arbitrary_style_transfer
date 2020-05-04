# Demo - train the style transfer network & use it to generate an image

from __future__ import print_function

from train import train
from infer import stylize
from utils import list_images
import argparse
import pdb

# arguments
parser = argparse.ArgumentParser(
    description='style transfer')
parser.add_argument('--content', type=str, nargs=1,
                    help='content path or dir relative to main folder', default=0)
parser.add_argument('--style', type=str, nargs=1,
                    help='style path or dir relative to main folder', default=0)                    
parser.add_argument('--reps', type=int, nargs=1,
                    help='number of recursive iterations for style loss', default=0)                    
args = parser.parse_args()
content_img = args.content[0]
style_img = args.style[0]
n_iter = args.reps[0]

IS_TRAINING = False

# for training
TRAINING_CONTENT_DIR = '../../fast-style-transfer/data/train2014'
TRAINING_STYLE_DIR = 'train'
ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'
LOGGING_PERIOD = 20

STYLE_WEIGHTS = [2.0]
MODEL_SAVE_PATHS = [
    # 'models/m9/style_weight_2e0_ep8.ckpt',
    'models/m10/style_weight_2e0_ep50.ckpt',
]

# for inferring (stylize)
INFERRING_CONTENT_DIR = 'images/content'
INFERRING_STYLE_DIR = 'images/style'
# OUTPUTS_DIR = 'outputs/m9_20epc_1it'
OUTPUTS_DIR = 'outputs'

# sorry for the hard coding but works
flag_cont = content_img[-4]=='.'
flag_style = style_img[-4]=='.'

def main():

    if IS_TRAINING:

        content_imgs_path = list_images(TRAINING_CONTENT_DIR)
        style_imgs_path   = list_images(TRAINING_STYLE_DIR)

        for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
            print('\n>>> Begin to train the network with the style weight: %.2f\n' % style_weight)

            train(style_weight, content_imgs_path, style_imgs_path, ENCODER_WEIGHTS_PATH, 
                  model_save_path, logging_period=LOGGING_PERIOD, debug=True)

        print('\n>>> Successfully! Done all training...\n')

    else:

        if not flag_cont:
            content_imgs_path = list_images(INFERRING_CONTENT_DIR)
        else:
            content_imgs_path = [content_img]
        if not flag_style:
            style_imgs_path   = list_images(INFERRING_STYLE_DIR)
        else:
            style_imgs_path = [style_img]

        for style_weight, model_save_path in zip(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
            print('\n>>> Begin to stylize images with style weight: %.2f\n' % style_weight)

            stylize(content_imgs_path, style_imgs_path, OUTPUTS_DIR, 
                    ENCODER_WEIGHTS_PATH, model_save_path, 
                    suffix='-' + str(style_weight), n_iter=n_iter)

        print('\n>>> Successfully! Done all stylizing...\n')


if __name__ == '__main__':
    main()

