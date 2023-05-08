import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse
from collections import OrderedDict

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer
from retinanet import model


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

	parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
	parser.add_argument('--coco_path', help='Path to COCO directory')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
	parser.add_argument('--annotation_delimiter', default = ' ')

	parser.add_argument('--model_path', help='Path to model (.pt) file.')

	parser = parser.parse_args(args)

	if parser.dataset == 'coco':
		dataset_val = CocoDataset(parser.coco_path, set_name='train2017', transform=transforms.Compose([Normalizer(), Resizer()]))
	elif parser.dataset == 'csv':
		dataset_val = CSVDataset(mode='Valid', train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]), \
			annotation_delimiter=parser.annotation_delimiter)
	else:
		raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    # Save folder
	save_folder = './weights'
	if not os.path.exists(save_folder):
		os.makedirs(save_folder)

	# Val results folder
	vis_results_folder = './vis_results'
	if not os.path.exists(vis_results_folder):
		os.makedirs(vis_results_folder)

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

	retinanet.eval()

	unnormalize = UnNormalizer()

	def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	for idx, data in enumerate(dataloader_val):

		if idx < 10:

			with torch.no_grad():
				st = time.time()
				if torch.cuda.is_available():
					scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
				else:
					scores, classification, transformed_anchors = retinanet(data['img'].float())
				
				print('Elapsed time: {}'.format(time.time()-st))
				idxs = np.where(scores.cpu()>0.5)
				img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

				img[img<0] = 0
				img[img>255] = 255

				img = np.transpose(img, (1, 2, 0))

				img_pred = cv2.cvtColor(img.copy().astype(np.uint8), cv2.COLOR_BGR2RGB)
				img_gt = cv2.cvtColor(img.copy().astype(np.uint8), cv2.COLOR_BGR2RGB)

				for j in range(idxs[0].shape[0]):
					bbox = transformed_anchors[idxs[0][j], :]
					x1 = int(bbox[0])
					y1 = int(bbox[1])
					x2 = int(bbox[2])
					y2 = int(bbox[3])
					label_name = dataset_val.labels[int(classification[idxs[0][j]])]
					draw_caption(img_pred, (x1, y1, x2, y2), label_name)

					cv2.rectangle(img_pred, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

				annot = data['annot'][0].cpu().numpy()
				for j in range(len(annot)):
					x1 = int(annot[j][0])
					y1 = int(annot[j][1])
					x2 = int(annot[j][2])
					y2 = int(annot[j][3])

					if x1 == -1:
						continue
						
					label_name = dataset_val.labels[int(annot[j][4])]
					draw_caption(img_gt, (x1, y1, x2, y2), label_name)

					cv2.rectangle(img_gt, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

				# cv2.imshow('img', img)
				img = cv2.imwrite(os.path.join(vis_results_folder, f'{idx}_predict.jpg'), img_pred)
				img = cv2.imwrite(os.path.join(vis_results_folder, f'{idx}_gt.jpg'), img_gt)
				print(f'Save {idx} img!')
				cv2.waitKey(0)

		else:
			break

if __name__ == '__main__':
 main()