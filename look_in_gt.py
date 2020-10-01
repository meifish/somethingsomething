import argparse
import json
import os
import sys
import itertools
import glob
from PIL import Image
import cv2
import numpy as np
import pprint


parser = argparse.ArgumentParser()
parser.add_argument('--gt_train_path', default='./gt_from_original_dataset/something-something-v2-train.json', type=str,
                    help='Path to the bounding box json')
parser.add_argument('--gt_validation_path', default='./gt_from_original_dataset/something-something-v2-validation.json', type=str,
                    help='Path to the bounding box json')
parser.add_argument('--gt_test_path', default='./gt_from_original_dataset/something-something-v2-test.json', type=str,    
                    help='Path to the bounding box json')
parser.add_argument('--gt_label_path', default='./gt_from_original_dataset/something-something-v2-labels.json', type=str,
                    help='Path to the bounding box json')
parser.add_argument('--id', default='1', type=str,
                    help='the id to look up')

if __name__ == '__main__':
    args = parser.parse_args()
    for i in [args.gt_train_path, args.gt_validation_path, args.gt_test_path]:
        with open(i, 'r') as gt:
            print("Finding in:", i)
            ground_truth = json.load(gt)

            id = '11309'
            for entry in ground_truth:            
                if args.id == entry['id']:
                    pprint.pprint(entry)
    
    sys.exit()