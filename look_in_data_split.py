import argparse
import json
import os
import sys
import pprint


parser = argparse.ArgumentParser()
parser.add_argument('--json', default='./code/dataset_splits/compositional/validation.json', type=str,
                    help='Path to the train data split jason')


if __name__ == '__main__':
    args = parser.parse_args()
    
    count = 0
    with open(args.json, 'r') as json_file:

        f_file = json.load(json_file)

        for i in f_file:
            if i['template'] == "Moving [something] up":
                count += 1

        print(count)
        sys.exit()
