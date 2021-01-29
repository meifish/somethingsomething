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

"""
python stats_dataset.py --objectA_path="./code/dataset_splits/compositional/ObjectsA.json" --objectB_path="./code/dataset_splits/compositional/ObjectsB.json" --splits_path="./code/dataset_splits/compositional/train.json"
"""


parser = argparse.ArgumentParser()
parser.add_argument('--objectA_path', default='./code/dataset_splits/compositional/ObjectsA.json', type=str,
                    help='Path to the ObjectA json.')
parser.add_argument('--objectB_path', default='./code/dataset_splits/compositional/ObjectsB.json', type=str,
                    help='Path to the ObjectB json.')
parser.add_argument('--splits_path', default='./code/dataset_splits/compositional/train.json', type=str,
                    help='Path to the data split json')
parser.add_argument('--annote_path1', default='./bounding_box_annotations/bounding_box_smthsmth_part1.json', type=str,
                    help='Path to the bounding box json')
parser.add_argument('--annote_path2', default='./bounding_box_annotations/bounding_box_smthsmth_part2.json', type=str,
                    help='Path to the bounding box json')
parser.add_argument('--annote_path3', default='./bounding_box_annotations/bounding_box_smthsmth_part3.json', type=str,    
                    help='Path to the bounding box json')
parser.add_argument('--annote_path4', default='./bounding_box_annotations/bounding_box_smthsmth_part4.json', type=str,
                    help='Path to the bounding box json')


if __name__ == '__main__':
    args = parser.parse_args()

    problem_log = open("missing.json", "w")
    problem_entry = {}

    with open(args.splits_path, 'r') as f:
        with open(args.objectA_path, 'r') as a:
            with open(args.objectB_path, 'r') as b:
                dataset = json.load(f)
                obj_A = json.load(a)
                obj_B = json.load(b)


                action_1 = [entry for entry in dataset if not all([obj in obj_A for obj in entry['new_placeholdes']])]
                action_2 = [entry for entry in dataset if not all([obj in obj_B for obj in entry['new_placeholdes']])]

                missing_obj = {}

                for i in (action_1+action_2):

                    try:
                        if '' in i['new_placeholdes']:
                            pprint.pprint(i)
                        
                            missing_obj[i['id']] = {'label': i['label'], 'template': i['template'], 'new_placeholdes': i['new_placeholdes'], 'placeholders': i['placeholders']}
                    except:
                        pass


                count = 0
                problem = 0
                for i in [args.annote_path1, args.annote_path2, args.annote_path3, args.annote_path4]:
                    with open(i, 'r') as anno:
                        print("Finding in:", i)
                        annotation = json.load(anno)

                        for j in annotation:

                            if j in missing_obj:

                                objects_anno = []
                                for frame in annotation[j]:
                                    obj_entries = frame["labels"]
                                    objects_anno += [obj["category"] for obj in obj_entries]
                                objects_anno = list(set(objects_anno))                                    

                                print("======")
                                print("missing object found")
                                print("id", j)
                                print("new_placeholdes", missing_obj[j]['new_placeholdes'])
                                print("placeholders", missing_obj[j]['placeholders'])
                                print("template", missing_obj[j]['template'])
                                print("label", missing_obj[j]['label'])
                                print("objects found in annotation:", objects_anno)
                                problem_entry[j] = missing_obj[j]
                                problem_entry[j]['objects_found_in_annotation'] = objects_anno

                                if len(objects_anno) == 0:
                                    problem += 1     
                                count += 1
                    
                json.dump(problem_entry, problem_log, indent=4)
                problem_log.close()

                print("re-identify missing objects:", count-problem)
                print("missing objects:", count)

                sys.exit()
