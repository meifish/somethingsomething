import argparse
import json
import os
import sys
import pprint


parser = argparse.ArgumentParser()
parser.add_argument('--annote', default='./annotation.json', type=str,
                    help='Path to the bounding box json')
parser.add_argument('--id', default='108042', type=str,
                    help='the id to look up')

if __name__ == '__main__':
    args = parser.parse_args()
    
    with open(args.annote, 'r') as anno:
        print("Finding in annotation:")
        annotation = json.load(anno)

        pprint.pprint(annotation[args.id])
    
        sys.exit()
