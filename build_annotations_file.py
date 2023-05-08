import argparse
import os
import numpy as np

def parsing_crowd_human_label(datalist):
	
	inputfile = {}   
	for i in np.arange(len(datalist)):
		adata = dict(eval(datalist[i].strip()))
		file_name = adata['ID']
		inputfile[file_name] = []
		gtboxes = adata['gtboxes']
		for gtbox in gtboxes:
			if gtbox['tag']=='person':
				data = {
				'name': 'person',
				'a head': gtbox['hbox'],
				'human visible-region': gtbox['vbox']
				}
				inputfile[file_name].append(data)
		
	return inputfile

def get_lines(data_path, inputfile):
	
	lines = []
	len_data = len(data_path)

	for i in range(len_data):

		img_path = data_path[i]
		img_name = img_path.split('/')[-1][:-4]
		annos = inputfile[img_name]
		
		objs = []
		for anno in annos:
			head_box = anno['a head']
			person_box = anno['human visible-region']

			person_x1 = person_box[0]
			person_y1 = person_box[1]
			person_x2 = person_x1 + person_box[2]
			person_y2 = person_y1 + person_box[3]

			head_x1 = head_box[0]
			head_y1 = head_box[1]
			head_x2 = head_x1 + head_box[2]
			head_y2 = head_y1 + head_box[3]

			objs.append([person_x1, person_y1, person_x2, person_y2, 'person'])
			objs.append([head_x1, head_y1, head_x2, head_y2, 'head'])

		for id, obj in enumerate(objs):
			# img_path x1 y1 x2 y2 class_name
			# delimiter = ' '
			lines.append(f'{img_path} {obj[0]} {obj[1]} {obj[2]} {obj[3]} {obj[4]}')
		
	return lines

def build_crowd_human_annotations(
	data_dir, \
	train_label_file, \
	valid_label_file):

	print('-'*30)
	print('Data dir:', data_dir)

	## Images
	train_data_path = []
	valid_data_path = []
	
	# Train
	data_dir_train1 = os.path.join(data_dir, 'CrowdHuman_train01', 'Images')
	for fname in os.listdir(data_dir_train1):
		train_data_path.append(os.path.join(data_dir, 'CrowdHuman_train01', 'Images', fname))
	data_dir_train2 = os.path.join(data_dir, 'CrowdHuman_train02', 'Images')
	for fname in os.listdir(data_dir_train2):
		train_data_path.append(os.path.join(data_dir, 'CrowdHuman_train02', 'Images', fname))
	data_dir_train3 = os.path.join(data_dir, 'CrowdHuman_train03', 'Images')
	for fname in os.listdir(data_dir_train3):
		train_data_path.append(os.path.join(data_dir, 'CrowdHuman_train03', 'Images', fname))
	print('Length of train images:', len(train_data_path))

	# Valid
	data_dir_valid = os.path.join(data_dir, 'CrowdHuman_val', 'Images')
	for fname in os.listdir(data_dir_valid):
		valid_data_path.append(os.path.join(data_dir, 'CrowdHuman_val', 'Images', fname))
	print('Length of valid images:', len(valid_data_path))

	## Labels
	
	# Train
	train_label_path = os.path.join(data_dir, train_label_file)
	print('Train label path:', train_label_path)
	with open(train_label_path, 'r+') as f:
		datalist = f.readlines()    
	train_inputfile = parsing_crowd_human_label(datalist)
	
	# Valid
	valid_label_path = os.path.join(data_dir, valid_label_file)
	print('Valid label path:', valid_label_path)
	with open(valid_label_path, 'r+') as f:
		datalist = f.readlines()        
	valid_inputfile = parsing_crowd_human_label(datalist)

	print('-'*30)

	## Build annotations csv file
	train_annotations_lines = get_lines(train_data_path, train_inputfile)
	print('Length of train annotations line:', len(train_annotations_lines))
	valid_annotations_line = get_lines(valid_data_path, valid_inputfile)
	print('Length of valid annotations line:', len(valid_annotations_line))
	
	# Train
	with open(os.path.join(data_dir, 'train_annotations.csv'), "w") as file:
		for line in train_annotations_lines:
			file.write(line + "\n")
		print('save train annotations csv file!')

	# Valid
	with open(os.path.join(data_dir, 'valid_annotations.csv'), "w") as file:
		for line in valid_annotations_line:
			file.write(line + "\n")
		print('save valid annotations csv file!')

def main(args):

	if args.dataset_name == 'crowd_human':
		build_crowd_human_annotations(
			data_dir='./data/CrowdHuman', \
			train_label_file='annotation_train.odgt', \
			valid_label_file='annotation_val.odgt', \
		)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--dataset_name", type=str, help="Path of class names txt file", choices=['crowd_human'])

    args = parser.parse_args()

    main(args)
