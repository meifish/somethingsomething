import argparse
import errno
import shutil
import json
import os
import sys
import pprint
import math
import random
from os.path import join


parser = argparse.ArgumentParser()
parser.add_argument('--annote', default='./annotation.json', type=str)   
parser.add_argument('--split_train_list', default='./code/dataset_splits/compositional/train.json', type=str)   
parser.add_argument('--split_validation_list', default='./code/dataset_splits/compositional/validation.json', type=str)
parser.add_argument('--labels', default='./code/dataset_splits/compositional/labels.json', type=str)
parser.add_argument('--frame_root', default='./something_videos_frames/', type=str)
parser.add_argument('--output_dir', default='./toy_dataset/', type=str)



def copy(src, dest):
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)


def clean_template(template):
    """ Replaces instances of `[something]` --> `something`"""
    template = template.replace("[", "")
    template = template.replace("]", "")
    return template


def count_indiv_action(split_list):
    labels_list = []
    
    # Save all action tempaltes into the list
    for i in range(len(split_list)):
        labels_list.append(clean_template(split_list[i]['template']))

    # Count num of each actions in train split             
    labels_list = sorted([(labels_list.count(action) , action) for action in set(labels_list)], key=lambda x: x[0])
    labels_count_dict = {action: count for (count, action) in labels_list}

    """
    {'action1' : count}
    """
    return labels_count_dict


def gen_sample_from_list(lables_count_dict, split_list, sample_ratio):
    sample_list = []
    
    sample_action_count = {k: math.floor(v*sample_ratio) for k,v in lables_count_dict.items() }

    for action in sample_action_count:
        pool = [d for d in split_list if clean_template(d['template']) == action]
        sample_size = sample_action_count[action]

        if sample_size > 0 :
            sample_list += random.sample(pool, sample_size)
    
    return sample_list


if __name__ == '__main__':
    args = parser.parse_args()
    

    anno_id_list = []    
    train_data_list = []
    validation_data_list = []
    labels_dict = {}
    anno_data_dict = {}


    sample_ratio = 0.01

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    compositional_path = join(args.output_dir, 'compositional')
    if not os.path.exists(compositional_path):
        os.mkdir(compositional_path)


    # Generate sample from train split 
    with open(args.split_train_list, 'r') as train_split:
        print('train....')
        train = json.load(train_split)
        
        # create count for each action
        action_count_dict = count_indiv_action(train)

        # sample from the train list
        train_data_list = gen_sample_from_list(action_count_dict, train, sample_ratio)
  
        # Add id for annotation, and train data split
        anno_id_list += [i['id'] for i in train_data_list]
            
        # Write to new file
        with open(join(args.output_dir, 'compositional', 'train.json'), 'w') as train_toy:
            json.dump(train_data_list, train_toy)


    # Generate sample from validation split 
    with open(args.split_validation_list, 'r') as validation_split:
        print('validate....')
        validate = json.load(validation_split)
        
        # create count for each action
        action_count_dict = count_indiv_action(validate)

        # sample from the validation list
        validation_data_list = gen_sample_from_list(action_count_dict, validate, sample_ratio)
  
        # Add id for annotation, and validation data split
        anno_id_list += [i['id'] for i in validation_data_list]

        # Write to new file
        with open(join(args.output_dir, 'compositional', 'validation.json'), 'w') as validate_toy:
            json.dump(validation_data_list, validate_toy)
    

    # Generate new annotation Json from the sample train and validation.
    with open(args.annote, 'r') as annotation:
        print('anno....')
        anno = json.load(annotation)

        for id in anno_id_list:
            anno_data_dict[id] = anno[id]

        # Write to new file
        with open(join(args.output_dir, 'annotation.json'), 'w') as anno_toy:
            json.dump(anno_data_dict, anno_toy)
    
    # Generate new label Json from the sample train and validation.
    with open(args.labels, 'r') as labels:
        print('labels....')
        labels = json.load(labels)

        total_sample = train_data_list + validation_data_list
        total_label = [clean_template(i['template']) for i in total_sample]
        total_label = list(set(total_label))
        labels_dict = {action: idx for (idx, action) in enumerate(total_label)}
        # label_count = sorted([(action, total_label.count(action)) for action in set(total_label)], key=lambda x: x[1])
        # labels_dict = {action: count for (action, count) in label_count}

        # Write to new file
        with open(join(args.output_dir, 'compositional', 'labels.json'), 'w') as label_toy:
            json.dump(labels_dict, label_toy)


    # Copy frames
    print("copy frames...")
    toy_frame_dir = join(args.output_dir, 'frames')
    if not os.path.exists(toy_frame_dir):
        os.mkdir(toy_frame_dir)

    for id in anno_id_list:
        source_video_path = join(args.frame_root, id)
        
        # walk in the indiv folder to get frame
        for x in os.walk(source_video_path):
            video_path = x[0]
            frames = x[2]

            video_base = os.path.basename(video_path)
            new_video_path = join(toy_frame_dir, video_base)
            os.mkdir(new_video_path)

            for f in frames:
                old_frame_path = join(video_path, f)
                new_frame_path = join(new_video_path, f)
                copy(old_frame_path, new_frame_path)



    sys.exit()
    