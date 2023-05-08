import argparse
import os
from collections import OrderedDict
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet import coco_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model', type=str)

    parser = parser.parse_args(args)

    # Val results folder
    val_results_folder = './val_results'
    if not os.path.exists(val_results_folder):
        os.makedirs(val_results_folder)

    dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    # Save folder
    save_folder = './weights'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Create the model
    retinanet = model.resnet50(num_classes=dataset_val.num_classes(),  model_dir=save_folder, pretrained=True)

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        state_dict = torch.load(parser.model_path)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.find('module') != -1:
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v

        retinanet.load_state_dict(new_state_dict)
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        state_dict = torch.load(parser.model_path)

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.find('module') != -1:
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v

        retinanet.load_state_dict(new_state_dict)
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    coco_eval.evaluate_coco(dataset_val, retinanet, val_results_folder)


if __name__ == '__main__':
    main()
