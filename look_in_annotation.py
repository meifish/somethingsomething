import argparse
import json
import os
import sys
import pprint


parser = argparse.ArgumentParser()
parser.add_argument('--annote_path1', default='./bounding_box_annotations/bounding_box_smthsmth_part1.json', type=str,
                    help='Path to the bounding box json')
parser.add_argument('--annote_path2', default='./bounding_box_annotations/bounding_box_smthsmth_part2.json', type=str,
                    help='Path to the bounding box json')
parser.add_argument('--annote_path3', default='./bounding_box_annotations/bounding_box_smthsmth_part3.json', type=str,    
                    help='Path to the bounding box json')
parser.add_argument('--annote_path4', default='./bounding_box_annotations/bounding_box_smthsmth_part4.json', type=str,
                    help='Path to the bounding box json')
parser.add_argument('--id', default='1', type=str,
                    help='the id to look up')

if __name__ == '__main__':
    args = parser.parse_args()
    for i in [args.annote_path1, args.annote_path2, args.annote_path3, args.annote_path4]:
        with open(i, 'r') as anno:
            print("Finding in:", i)
            annotation = json.load(anno)

            if args.id in annotation:
                pprint.pprint(annotation[id])
