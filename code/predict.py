import argparse
import json
import os
import numpy as np
import pickle
from pathlib import Path
import torch
from callbacks import AverageMeter
from data_utils.data_loader_frames import VideoFolder
from train import accuracy
"""
Description: This predict module only works for pre-annonated val dataset of something-something-v2
             i.e., with bounding box annotations)
 
python ./code/predict.py    --json_data_list ./code/dataset_splits/compositional/validation.json 
                            --json_file_labels ./code/dataset_splits/compositional/labels.json
                            --tracked_boxes ./annotation.json
                            --model coordAttention
                            --model_state_dict_path ./ckpt/test_exp_Attention_fixbug_fulldata_best.pth.tar
                            --root_frames ./something_videos_frames/
                            --result_path ./prediction_attention_result.json
                            
"""

parser = argparse.ArgumentParser(description='PyTorch Smth-Else Predictor')

##################
# args: JSONs
##################
parser.add_argument('--json_data_list', default='', type=str, 
                    help='path to the json file with either train or validation video meta data')
parser.add_argument('--json_file_labels',  default='', type=str, 
                    help='path to the json file with ground truth labels')
parser.add_argument('--word2vec_weights_path', default='', type=str, 
                    help='path to the word2vec pre-trained weight')
parser.add_argument('--tracked_boxes', type=str, help='path to the tracked box annotation json')

##################
# args: Model
##################
parser.add_argument('--model',
                    default='coord')
parser.add_argument('--model_state_dict_path', type=str,
                    help='state dict model path that will be used for inference')
parser.add_argument('--num_boxes', default=4, type=int,
                    help='num of boxes for each image')
parser.add_argument('--num_classes', default=174, type=int,
                    help='num of class in the model')
parser.add_argument('--coord_feature_dim', default=256, type=int, metavar='N',       #TODO: pull out from the loaded model
                    help='intermediate feature dimension for coord-based features')
parser.add_argument('--batch_size', '-b', default=72, type=int,
                    metavar='N', help='mini-batch size (default: 72)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--fine_tune', help='path with ckpt to restore')

##################
# args: Video Frames
##################
parser.add_argument('--video_root', default='./something_videos/', type=str, help='path to the folder with videos')
parser.add_argument('--root_frames', type=str, help='path to the folder with frames')
parser.add_argument('--num_frames', default=16, type=int,
                    help='num of frames for the model. (Needs to match the loaded model trained number of frame.)')

##################
# args: Result
##################
parser.add_argument('--result_path', type=str, 
                    help='path to the JSON file path to save the prediction result')

args = parser.parse_args()
cuda_device = torch.device(0)



def predict(model, result_file_path=args.result_path):
    global args
    
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    # Pickle path for intermediate batch result:
    temp_pickle = os.path.splitext(result_file_path)[0] + '.p'
    if os.path.exists(temp_pickle):
        os.remove(temp_pickle)

    #model = torch.nn.DataParallel(model)
    
    # Read Model in
    checkpoint = torch.load(args.model_state_dict_path)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    
    # Set eval mode
    model.eval()

    # Data Loader
    dataset = VideoFolder(
                        root=args.root_frames,
                        num_boxes=args.num_boxes,
                        file_input=args.json_data_list,
                        file_labels=args.json_file_labels,
                        word2vec_weights=args.word2vec_weights_path, 
                        frames_duration=args.num_frames,
                        video_root=args.video_root,
                        args=args,
                        is_val=True,
                        if_augment=True,
                        model=args.model
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True,
        pin_memory=True
    )
    



    for i, (vid_names, frame_list, global_img_tensors, box_tensors, box_categories, word2vec_features, video_label) in enumerate(data_loader):    

        # Move data to GPU
        global_img_tensors = global_img_tensors.to(cuda_device)
        box_categories = box_categories.to(cuda_device)
        box_tensors = box_tensors.to(cuda_device)
        video_label = video_label.to(cuda_device)
        if not isinstance(word2vec_features, list):       # word2vec has dummpy [] holder if no word2vec provided
            word2vec_features = word2vec_features.to(cuda_device)

        with torch.no_grad():
            output = model(global_img_tensors, box_categories, box_tensors, word2vec_features, video_label)
            attention = model.batch_attention_weight
            print("Atten shape", attention.size())
            output = output.view((-1, len(data_loader.dataset.classes)))

            [acc1, acc5], pred = accuracy(output, video_label, topk=(1, 5), return_predict=True)
            acc_top1.update(acc1.item(), global_img_tensors.size(0))
            acc_top5.update(acc5.item(), global_img_tensors.size(0))

            with open(temp_pickle, 'ab+') as fp:
                
                result_dict = {
                    'vid_names' : vid_names,
                    'frame_list' : frame_list.int().tolist(),          # Convert tensor to int type and then to list.
                    'video_label' : video_label.int().cpu().numpy(),  
                    'prediction' : pred.int().cpu().numpy(),
                    'attention' : attention.cpu().numpy()
                } 

                pickle.dump(result_dict, fp)
            
            batch_result = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Acc1 {acc_top1.val:.1f} ({acc_top1.avg:.1f})\t' \
                    'Acc5 {acc_top5.val:.1f} ({acc_top5.avg:.1f})'.format(
                    epoch, i, len(data_loader), acc_top1=acc_top1, acc_top5=acc_top5)

            print(batch_result)
    
    aggre_batch_result(temp_pickle, result_file_path=result_file_path)



def aggre_batch_result(batch_result_pickle, result_file_path=args.result_path):
    result_dict = {}
    result_dict['vid_names'] = []  # List of string vid, i.e., '183130'.
    result_dict['frame_list'] = []
    result_dict['video_label'] = np.array([], dtype=int)
    result_dict['prediction'] = np.empty([0, 5], dtype=int)
    result_dict['attention'] = np.empty((0, 4, 8, 4), dtype=int)  # (b, nr_boxes, frame, nr_boxes)

    # Iteratively read in batch result saved in pickle.
    with open(batch_result_pickle, 'rb') as fr:
        try:
            while True:
                batch_result = pickle.load(fr)
                result_dict['vid_names'].extend(batch_result['vid_names'])
                result_dict['frame_list'].extend(batch_result['frame_list'])
                result_dict['video_label'] = np.append(result_dict['video_label'], batch_result['video_label'], axis=0)
                result_dict['prediction'] = np.append(result_dict['prediction'], batch_result['prediction'], axis=0)
                result_dict['attention'] = np.append(result_dict['attention'], batch_result['attention'], axis=0)

        except EOFError:
            pass

    result_dict['video_label'] = result_dict['video_label'].tolist()
    result_dict['prediction'] = result_dict['prediction'].tolist()
    result_dict['attention'] = result_dict['attention'].tolist()


    # Save the aggregate to the result JSON
    if os.path.splitext(result_file_path)[1] == '.json':
        with open(result_file_path, 'w') as fw:
            
                json.dump(result_dict, fw)
                os.remove(batch_result_pickle)
        
    elif os.path.splitext(result_file_path)[1] == '.p':
        with open(result_file_path, 'wb') as fw:
            
                pickle.dump(result_dict, fw)


    return result_dict


def read_dict(result_file_path=args.result_path):
    with open(result_file_path, 'r') as fr:
        result_dict = json.load(fr)
    return result_dict


def main():
    global args

    if args.model == 'coord':
        from model.model_lib import VideoModelCoord as VideoModel
    elif args.model == 'coordAttention':
        from model.model_lib import VideoModelCoordAttention as VideoModel
    elif args.model == 'coordAdd':
        from model.model_lib import VideoModelCoordAdd as VideoModel

    # model = VideoModel(args)
    # model.to(cuda_device)
    # result_dict = predict(model)

    result_dict = read_dict()
    print(result_dict['vid_names'][:100])
    
    print("Result is saved.")


if __name__ == '__main__':
    main()