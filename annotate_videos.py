import argparse
import json
import os
import itertools
import glob
from PIL import Image
import cv2
import numpy as np
import pprint
import math


"""
python annotate_videos.py --vid_path ./something_videos_frames_to_annotate/
                          --out_vid_path ./annotated_videos/ 
                          --annot_path=./annotation.json
                          --json_file_labels ./code/dataset_splits/compositional/labels.json
                          --prediction_path=./prediction_attention_result.json

"""
parser = argparse.ArgumentParser()
parser.add_argument('--vid_path', default='videos/', type=str,
                    help='Path to the directory with videos')
parser.add_argument('--out_vid_path', default='annotated_videos/', type=str,
                    help='Path to the directory where the annotated videos are saved')
parser.add_argument('--annot_path', default='annotations.json', type=str,
                    help='Path to the box annotation file')
parser.add_argument('--json_file_labels',  default='', type=str, 
                    help='path to the json file with ground truth labels')
parser.add_argument('--annotate_objects', default=False, type=bool,
                    help='Flag indicating whether to annotate the bounding boxes with names of objects')
parser.add_argument('--prediction_path', type=str,
                    help='Path to the prediction file.')



COLORS = [(255, 64, 64), (0, 0, 255), (127, 255, 0), (255, 97, 3), (220, 20, 60),
          (255, 185, 15), (255, 20, 147), (255, 105, 180), (60, 179, 113)]


HLS_COLOR = [(0, 128, 255), (60, 85, 255), (120, 102, 255), (135, 128, 255), (25, 128, 255)] # RED, GREEN, BLUE, PURPLE, YELLOW


def read_dict(args):
    """
        Convert the prediction result to video index dictionary
        {vid : {'video_label': 134, 
                'prediction' : [3,1,5,6,4],
                'attention'  : [[],[],...]}  #shape: (n_box, frames, n_box)
        }
    """

    with open(args.prediction_path) as fr:
        result_dict = json.load(fr)
        result_dict_return = {}
        for i,v in enumerate(result_dict['vid_names']):
            
            if v in ['149', '28','49', 149, 28, 49]:
                print(f'Target {v}, {type(v)} does in prediction!!! Why lying!!')
            result_dict_return[v] = {}
            result_dict_return[v]['video_label'] = result_dict['video_label'][i]
            result_dict_return[v]['frame_list'] = result_dict['frame_list'][i]
            result_dict_return[v]['prediction'] = result_dict['prediction'][i]
            result_dict_return[v]['attention'] = result_dict['attention'][i]

        # result_dict = {v: {key,value} for key,value in \
        #     zip(('video_label', 'frame_list', 'prediction', 'attention'), \
        #         (result_dict['video_label'][i], result_dict['frame_list'][i], result_dict['prediction'][i], result_dict['attention'][i])) \
        #         for i,v in enumerate(result_dict['vid_names'])}
    
    return result_dict_return


def read_label(args):

    with open(args.json_file_labels) as fr:
        label_dict = json.load(fr)
        label_dict = {v:k for k,v in label_dict.items()}

    return label_dict
    

def annotate_frame(meta, args, object_set, target_box, attention, gt_pred_labels):
    frame_path = os.path.join(args.vid_path, meta['name'])
    image = Image.open(frame_path)
    im_height = image.size[1]
    image = np.array(image, dtype=np.uint8)

    target_c = HLS_COLOR[target_box]

    print(list(object_set))
    print(attention)


    for i in meta['labels']:
        x1, x2, y1, y2 = i['box2d']['x1'], i['box2d']['x2'], i['box2d']['y1'], i['box2d']['y2']
        
        standard_category = i['standard_category']
        global_box_id = object_set.index(standard_category)
        attention_weight = attention[global_box_id]
        
        print('target box:{0}\t standard category:{1} \t global box:{2}'.format(target_box, standard_category, global_box_id))

        # Pick the HSL color based on the attention weight
        if global_box_id == target_box:
            color = target_c
            print(1)
            print("color:", color)
        else:
            color = (target_c[0], target_c[1] + (255-target_c[1]) * (1-attention[global_box_id]), target_c[2])
            print(attention_weight)
            print("color:", color)

        color = cv2.cvtColor( np.uint8([[color]] ), cv2.COLOR_HLS2RGB)[0][0]
        color = tuple(map(int, color))

        # Draw boxes
        image = cv2.rectangle(
            image, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
        
        # Annotate category
        if args.annotate_objects:
            cv2.putText(image, i['category'], (int(x1)+5, int(y1)+20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)

        # Annotate weights   
        if attention_weight != 0:
            cv2.putText(image, "%.4f" % round(attention_weight, 4), (int(x1)+5, int(y1)+20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
    

    # Annotate labels
    labels = gt_pred_labels.copy()
    for i, label in enumerate(labels):
        chunk_len = 40
        if len(label) > chunk_len:
            labels[i] = '\n'.join(list(map(''.join, zip(*[iter(label)]*chunk_len))))
            labels[i] += '\n' + label[ (len(label)//chunk_len)*chunk_len : ]
            print('labels:', labels[i])

    y, dy = im_height-60, 20
    colors = [(255, 0, 0), (0, 255, 0)] # Ground Truth, Predicted

    for l, label in enumerate(labels):
        print(label.split('\n'))
        for i, text in enumerate(label.split('\n')):
            #print(text, y)
            cv2.putText(image,  text, (0, y), cv2.FONT_HERSHEY_DUPLEX, 0.6, colors[l], 1) 
            y+=dy
            print(y)

    im = Image.fromarray(image)

    vid = os.path.split(meta['name'])[0]
    frame = os.path.split(meta['name'])[1]
    print(os.path.split(meta['name']))
    vid_path = os.path.join(args.out_vid_path, vid, str(target_box))
    os.makedirs(vid_path, exist_ok=True)
    im.save(os.path.join(vid_path, frame))



def get_colormap(meta):
    all_objects = [[j['category'] for j in i['labels']] for i in meta]
    all_objects = np.unique(list(itertools.chain.from_iterable(all_objects)))
    color_map = {all_objects[i]: COLORS[i] for i in range(len(all_objects))}
    return color_map



def annotate_video(video_path, annotations, prediction, text_labels, args):
    vid_id = os.path.basename(video_path).split('.')[0]
    try:
        meta = annotations[vid_id]
    except:
        print('Annotations for the video {} not found, skipping!'.format(vid_id))
        return
    try:
        predict = prediction[vid_id]
    except:
        print('Predictions for the video {} not found, skipping!'.format(vid_id))
        return
    

    os.makedirs(os.path.join(args.out_vid_path, vid_id), exist_ok=True)
    color_map = get_colormap(meta)
    
    # Retreive text labels
    pred_label = text_labels[predict['prediction'][0]]
    gt_label = text_labels[predict['video_label']]
    labels = [gt_label, pred_label]
    

    # Retrieve global box id across all frames
    object_set = set()
    for frame_idx in predict['frame_list']:        
        meta_frame = meta[frame_idx]

        for box_data in meta_frame['labels']:
            standard_category = box_data['standard_category']
            object_set.add(standard_category)
    object_set = sorted(list(object_set))
    
    # Start annotating frames per each target box.
    for target_box in range(4):
        #for meta_frame in meta:
        for i, frame_idx in enumerate(predict['frame_list']):
            meta_frame = meta[frame_idx]
            pred_label = str(predict['prediction'][0])   # Needs to convert these numeral labels to text label.
            true_label = str(predict['video_label'])
            attention = predict['attention'][target_box][i]
            annotate_frame(meta_frame, args, object_set, target_box, attention, labels)


if __name__ == '__main__':
    args = parser.parse_args()

    with open(args.annot_path, 'r') as f:
        annotations = json.load(f)

    predictions = read_dict(args)
    text_labels = read_label(args)

    os.makedirs(args.out_vid_path, exist_ok=True)

    video_paths = glob.glob(args.vid_path + '/*')
    for video_path in video_paths:
        annotate_video(video_path, annotations, predictions, text_labels, args)