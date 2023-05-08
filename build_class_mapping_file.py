import argparse
import os

def main(args):
	print('args:', args)
	
	with open(args.class_names_txt, "r") as file:
	    names = file.readlines()
	 
	print('names:', names)
	print()
	
	names_csv_lines = []
	for i, name in enumerate(names):
	    names_csv_lines.append(f'{name.strip()},{i}')
	
	print('names_csv_lines:', names_csv_lines)
	
	with open(args.class_names_csv, "w") as file:
		for names_csv_line in names_csv_lines:
			file.write(names_csv_line + "\n")
		print('save csv file!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-t", "--class_names_txt", type=str, help="Path of class names txt file", \
    	default = './class_names/coco_names_with_head.txt')
    parser.add_argument("-c", "--class_names_csv", type=str, help="Path of class name csv file", \
    	default = './class_names/coco_names_with_head.csv')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
