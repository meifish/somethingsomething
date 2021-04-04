import argparse
import collections
import json
import os
import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import heatmap as plt_hmp

parser = argparse.ArgumentParser(description='PyTorch Smth-Else Predictor')
parser.add_argument('--json_result_dict', type=str, 
                    help='path to the json file produced by predict.py')
parser.add_argument('--json_file_labels', type=str, help='path to the json file with ground truth labels')
parser.add_argument('--num_classes', default=174, type=int,
                    help='num of class in the model')

args = parser.parse_args()

"""
python ./code/result_analysis/analysis.py    --json_result_dict ./prediction_result.p
                                             --json_file_labels ./toy_dataset/compositional/labels.json

"""

def read_result_dict(result_dict_path, pickle_file=False):
    
    result_dict= {}
    with open(result_dict_path, 'rb') as fr:
        if pickle_file == True:
            result_dict = pickle.load(fr)
        else:
            result_dict = json.load(fr)
                
    return result_dict


def analyze_confused5 (result_dic_path=args.json_result_dict):
   
    #############################
    # Prediction Result
    #############################
    result = read_result_dict(result_dic_path, pickle_file=True)

    # Numpy array
    gt = result['video_label']   # (b, )
    pred = result['prediction']  # (b, 5) top 5 prediction

    unique, counts = np.unique(gt, return_counts=True)
    label_count = dict(zip(unique, counts))

    #############################
    # Prediction Result
    #############################
    """
    pred_err_rows: wrong prediction (i.e., corrent label does not equal to prediction)
    confused_rows: confused prediction (i.e., correct label is in top 5 prediction)

    """
    gt = np.expand_dims(gt, -1)
    gt = np.repeat(gt, 5, 1)

    # Err Rows
    pred_err_rows = (gt[:, 0] != pred[:, 0]).sum()
    
    # Confused Rows
    confused_rows = (gt[:, 1:] == pred[:, 1:]).sum(axis=1).astype(bool)
    confused_pred = pred[confused_rows, 0]
    confused_gt = gt[confused_rows, 0]

    confusion_dict = list(zip(confused_gt, confused_pred))
    confusion_dict = collections.Counter(confusion_dict)
    

    
    print('Total prediction err: {0}\t'
          'Total confusion err: {1}\t'
        .format(pred_err_rows, confused_rows.sum()))


    # Establish heatmap matrix to plot
    confusion_matrix = np.zeros((args.num_classes, args.num_classes))
    for gt in range(args.num_classes):
        
        num_confused_per_gt = sum([val for key, val in confusion_dict.items() if key[0]==gt])
        
        for pred in range(args.num_classes):
            if (gt,pred) in confusion_dict:
                             
                confusion_matrix[gt, pred] = confusion_dict[(gt,pred)] / num_confused_per_gt * 100


    with open(args.json_file_labels, 'rb') as fr:
        catg_dict = json.load(fr)

    category_labels = [l[0] for l in sorted(catg_dict.items(), key=lambda item: item[1])]    
   

    # Plot
    grid = 3
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    
    
    for i in range(grid):
        for j in range(grid):

            section_size = int(args.num_classes/grid)
            x_range = [i * section_size, (i+1) * section_size]
            y_range = [j * section_size, (j+1) * section_size]
            grid_matrix = confusion_matrix[y_range[0] : y_range[1], x_range[0] : x_range[1]]
            
            x_category = category_labels[x_range[0]: x_range[1]]
            y_category = category_labels[y_range[0]: y_range[1]]

            im, cbar = plt_hmp.heatmap(grid_matrix, y_category, x_category, ax=ax,
                            cmap="YlGn",
                            vmin=0, vmax=100,
                            cbarlabel="Top 5 Confusion Percentage Per Category")
    
            fig.tight_layout()
            plt.savefig(f'heatmap_{i}_{j}.png')
            cbar.remove()


    # im, cbar = plt_hmp.heatmap(confusion_matrix, category_labels, category_labels, ax=ax,
    #                 cmap="YlGn",
    #                 vmin=0, vmax=40,
    #                 cbarlabel="Top 5 Confusion Percentage Per Category")            
    # fig.tight_layout()
    # plt.savefig(f'heatmap_complete.png')
    



def main():
    global args
    

    analyze_confused5()


if __name__ == '__main__':
    main()