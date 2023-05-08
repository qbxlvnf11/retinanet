import argparse
import os
from collections import OrderedDict
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_val', help='Path to CSV annotations')
    parser.add_argument('--model_path', help='Path to model', type=str)
    # parser.add_argument('--images_path',help='Path to images directory',type=str)
    parser.add_argument('--csv_classes',help='Path to classlist csv',type=str)
    parser.add_argument('--annotation_delimiter', default = ' ')
    parser.add_argument('--iou_threshold',help='IOU threshold used for evaluation',type=str, default='0.5')
    parser = parser.parse_args(args)

    #dataset_val = CocoDataset(parser.coco_path, set_name='val2017',transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CSVDataset(mode='Valid', train_file=parser.csv_val, class_list=parser.csv_classes, \
        transform=transforms.Compose([Normalizer(), Resizer()]), \
        annotation_delimiter=parser.annotation_delimiter)

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

    print(csv_eval.evaluate(dataset_val, retinanet,iou_threshold=float(parser.iou_threshold)))

if __name__ == '__main__':
    main()
