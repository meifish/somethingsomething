import argparse
import os
from pathlib import Path
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
import torchvision
from callbacks import AverageMeter, Logger
from data_utils.data_loader_frames import VideoFolder
from utils import save_results
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch Smth-Else')

# Path related arguments


"""
python ./code/train.py --coord_feature_dim 256 --root_frames ./toy_dataset/
                        --json_data_train ./toy_dataset/compositional/train.json
                        --json_data_val ./toy_dataset/compositional/validation.json 
                        --json_file_labels ./toy_dataset/compositional/labels.json
                        --word2vec_weights_path ./code/data_utils/word2vec/GoogleNews-vectors-negative300.bin
                        --tracked_boxes ./toy_dataset/annotation.json
                        --model coordSemDualAttention
                        --logname exp_dualAttention_toydata

python ./code/train.py --coord_feature_dim 256 --root_frames ./something_videos_frames/
                        --json_data_train ./code/dataset_splits/compositional/train.json
                        --json_data_val ./code/dataset_splits/compositional/validation.json 
                        --json_file_labels ./code/dataset_splits/compositional/labels.json
                        --tracked_boxes ./annotation.json
                        --model coordAttention
                        --logname exp_Attention_fixbug_fulldata
                         

"""

parser.add_argument('--model',
                    default='coord')
parser.add_argument('--root_frames', type=str, help='path to the folder with frames')
parser.add_argument('--video_root', default='./something_videos/', type=str, help='path to the folder with videos')
parser.add_argument('--json_data_train', type=str, help='path to the json file with train video meta data')
parser.add_argument('--json_data_val', type=str, help='path to the json file with validation video meta data')
parser.add_argument('--json_file_labels', type=str, help='path to the json file with ground truth labels')
parser.add_argument('--word2vec_weights_path', default='', type=str, help='path to the word2vec pre-trained weight')
parser.add_argument('--img_feature_dim', default=256, type=int, metavar='N',
                    help='intermediate feature dimension for image-based features')
parser.add_argument('--coord_feature_dim', default=128, type=int, metavar='N',
                    help='intermediate feature dimension for coord-based features')
parser.add_argument('--clip_gradient', '-cg', default=5, type=float,
                    metavar='W', help='gradient norm clipping (default: 5)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--size', default=224, type=int, metavar='N',
                    help='primary image input size')
parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '-b', default=72, type=int,
                    metavar='N', help='mini-batch size (default: 72)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[24, 35, 45], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print_freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--log_freq', '-l', default=1, type=int,
                    metavar='N', help='frequency to write in tensorboard (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--num_classes', default=174, type=int,
                    help='num of class in the model')
parser.add_argument('--num_boxes', default=4, type=int,
                    help='num of boxes for each image')
parser.add_argument('--num_frames', default=16, type=int, # orginally 4
                    help='num of frames for the model')
parser.add_argument('--dataset', default='smth_smth',
                    help='which dataset to train')
parser.add_argument('--acc_history_dir', default='./accu',
                    help='folder to output accuracy logsper epoch')
parser.add_argument('--logdir', default='./logs',
                    help='folder to output tensorboard logs')
parser.add_argument('--logname', default='exp',
                    help='name of the experiment for checkpoints and logs')
parser.add_argument('--ckpt', default='./ckpt',
                    help='folder to output checkpoints')
parser.add_argument('--arch', default='test',
                    help='experiment name')
parser.add_argument('--fine_tune', help='path with ckpt to restore')
parser.add_argument('--tracked_boxes', type=str, help='choose tracked boxes')
parser.add_argument('--shot', default=5)
parser.add_argument('--restore_i3d')
parser.add_argument('--restore_custom')

best_loss = 1000000
writer = SummaryWriter('./runs')
cuda_device = torch.device(0)



def check_gpu(msg='gpu_check'):
    print(f'{msg:=^60}')
    # print('Memory Usage:')
    print(f'Allocated:, {round(torch.cuda.memory_allocated(0)/1024**3,6)}GB')
    print(f'Allocated:, {round(torch.cuda.memory_allocated(1)/1024**3,6)}GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(1)/1024**3,1), 'GB')

def main():
    
    global args, best_loss
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        os.mkdir(args.ckpt)

    print(args)


    # create model
    if args.model == 'coord':
        from model.model_lib import VideoModelCoord as VideoModel
    elif args.model == 'coordAttention':
        from model.model_lib import VideoModelCoordAttention as VideoModel
    elif args.model == 'coordAdd':
        from model.model_lib import VideoModelCoordAdd as VideoModel
    elif args.model == 'coordSemDualAttention':
        from model.model_lib import VideoModelCoorSemDualdAttention as VideoModel
    elif args.model == 'coord_latent':
        from model.model_lib import VideoModelCoordLatent as VideoModel
    elif args.model == 'coord_latent_nl':
        from model.model_lib import VideoModelCoordLatentNL as VideoModel
    elif args.model == 'global_coord_latent':
        from model.model_lib import VideoModelGlobalCoordLatent as VideoModel
    elif args.model == 'global_coord_latent_nl':
        from model.model_lib import VideoModelGlobalCoordLatentNL as VideoModel
    elif args.model == 'global_i3d':
        from model.model_lib import VideoGlobalModel as VideoModel
    elif args.model == 'global_coord':
        from model.model_lib import VideoModelGlobalCoord as VideoModel
    model = VideoModel(args)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "No checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        if args.start_epoch is None:
            args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))

    if args.start_epoch is None:
        args.start_epoch = 0

    model.to(cuda_device)
    #model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

    # create training and validation dataset
    dataset_train = VideoFolder(root=args.root_frames,
                                num_boxes=args.num_boxes,
                                file_input=args.json_data_train,
                                file_labels=args.json_file_labels,
                                word2vec_weights=args.word2vec_weights_path,
                                frames_duration=args.num_frames,
                                video_root=args.video_root,
                                args=args,
                                is_val=False,
                                if_augment=True,
                                model=args.model,
                                )

   
    dataset_val = VideoFolder(root=args.root_frames,
                              num_boxes=args.num_boxes,
                              file_input=args.json_data_val,
                              file_labels=args.json_file_labels,
                              word2vec_weights=args.word2vec_weights_path, 
                              frames_duration=args.num_frames,
                              video_root=args.video_root,
                              args=args,
                              is_val=True,
                              if_augment=True,
                              model=args.model,
                              )


    # create training and validation loader
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True,
        pin_memory=True
    )

    
    val_loader = torch.utils.data.DataLoader(
        dataset_val, drop_last=True,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    optimizer = torch.optim.SGD(model.parameters(), momentum=args.momentum,
                                lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    if args.evaluate:
        validate(val_loader, model, criterion, class_to_idx=dataset_val.classes_dict)
        return

    # training, start a logger
    tb_logdir = os.path.join(args.logdir, args.logname)

    if not (os.path.exists(tb_logdir)):
        os.makedirs(tb_logdir)
    tb_logger = Logger(tb_logdir)

    acc_history = {}

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        loss, acc_top1, acc_top5 = train(train_loader, model, optimizer, epoch, criterion, tb_logger)
        acc_history["acc_top1_epoch_training"] = (epoch, acc_top1)
        acc_history["acc_top5_epoch_training"] = (epoch, acc_top5)
        acc_history["loss_epoch_training"] = (epoch, loss)

        # evaluate on validation set
        # if (not args.fine_tune) or (epoch + 1) % 10 == 0:
        if (not args.fine_tune):
            loss, acc_top1, acc_top5 = validate(val_loader, model, criterion, epoch=epoch, tb_logger=tb_logger)
            acc_history["acc_top1_epoch_val"] = (epoch, acc_top1)
            acc_history["acc_top5_epoch_val"] = (epoch, acc_top5)
            acc_history["loss_epoch_val"] = (epoch, loss)
        else:
            loss = 100

        # remember best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': args.arch,
                #'state_dict': model.module.state_dict(),
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
            },
            is_best,
            os.path.join(args.ckpt, args.arch.lower() + '_{}'.format(args.logname)))


        # Finish one epoch, log the acc
        accu_path = Path(args.acc_history_dir) / args.logname
        if not os.path.exists(accu_path):
            accu_path.mkdir(parents=True)
        
        for key in acc_history:          
            acc_file = f"{key}.txt"
            acc_file = accu_path / acc_file

            if os.path.exists(acc_file):
                append_write = 'a' 
            else:
                append_write = 'w' 

            with open(acc_file, append_write) as f:
                f.write(str(acc_history[key][0]) + "\t" + str(acc_history[key][1]) + "\n")



def train(train_loader, model, optimizer, epoch, criterion, tb_logger=None):
    global args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()

    # switch to train mode
    model.train()
   
    end = time.time()
    # check_gpu('Start')
    #for i, (vid_names, frame_tensors, global_img_tensors, box_tensors, box_categories, word2vec_features, video_label) in enumerate(train_loader):
    for i, (vid_names, frame_list, global_img_tensors, box_tensors, box_categories, word2vec_features, video_label) in enumerate(train_loader):    
        
        ################################################################
        # output to tensorboard
        # print("print frame shape:", frame_tensors.shape)  #[72,8,3,224,224]
        # print("len vid_names:", len(vid_names))

        #for v, video in enumerate(frame_tensors): 

            #print("video shape:", video.shape)         #[8, 3, H, W]
            #frames = [frame for frame in video]       #[8, 3, H, W]
            #print("frames shape:", frames[0].shape)   #[3, H, W]
            
            #img_grid = torchvision.utils.make_grid(video, nrow=8)
            #print("image grid shape:", img_grid.shape)  #[3, H, 8*W]
            #transposed_frame = np.transpose(img_grid.numpy(), (1, 2, 0))
            #print("transposed shape:", transposed_frame)
            #plt.imshow(transposed_frame)
          
            #writer.add_images(vid_names[v], video)
        ################################################################

        model.zero_grad()
        # measure data loading time
        data_time.update(time.time() - end)

        # local_img_tensor is (b, nr_frames, nr_boxes, 3, h, w)
        # global_img_tensor is (b, nr_frames, 3, h, w)

        # compute output
        global_img_tensors = global_img_tensors.to(cuda_device)
        box_categories = box_categories.to(cuda_device)
        box_tensors = box_tensors.to(cuda_device)
        video_label = video_label.to(cuda_device)
        
        #print("type w2v:", word2vec_features)
        if not isinstance(word2vec_features, list):  # word2vec has dummpy [] holder if no word2vec provided
           word2vec_features = word2vec_features.to(cuda_device)

        output = model(global_img_tensors, box_categories, box_tensors, word2vec_features, video_label)
        output = output.view((-1, len(train_loader.dataset.classes)))
        # check_gpu('after train')

        #loss = criterion(output, video_label.long().cuda())
        loss = criterion(output, video_label)
        # check_gpu('after loss')

        acc1, acc5 = accuracy(output.cpu(), video_label.cpu(), topk=(1, 5))

        # measure accuracy and record loss
        losses.update(loss.item(), global_img_tensors.size(0))
        acc_top1.update(acc1.item(), global_img_tensors.size(0))
        acc_top5.update(acc5.item(), global_img_tensors.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            batch_result = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                  'Acc1 {acc_top1.val:.1f} ({acc_top1.avg:.1f})\t' \
                  'Acc5 {acc_top5.val:.1f} ({acc_top5.avg:.1f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, acc_top1=acc_top1, acc_top5=acc_top5)
            print(batch_result)


        # log training data into tensorboard
        if tb_logger is not None and i % args.log_freq == 0:
            logs = OrderedDict()
            logs['Train_IterLoss'] = losses.val
            logs['Train_Acc@1'] = acc_top1.val
            logs['Train_Acc@5'] = acc_top5.val
            # how many iterations we have trained
            iter_count = epoch * len(train_loader) + i
            for key, value in logs.items():
                tb_logger.log_scalar(value, key, iter_count)

            tb_logger.flush()
        
        
    # Finish training for one epoch
    return losses.avg, acc_top1.avg, acc_top5.avg


def validate(val_loader, model, criterion, epoch=None, tb_logger=None, class_to_idx=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    logits_matrix = []
    targets_list = []
    # switch to evaluate mode
    model.eval()

    end = time.time()
    #for i, (vid_names, frame_tensors, global_img_tensors, box_tensors, box_categories, word2vec_features, video_label) in enumerate(val_loader):
    for i, (vid_names, frame_list, global_img_tensors, box_tensors, box_categories, word2vec_features, video_label) in enumerate(val_loader):

        global_img_tensors = global_img_tensors.to(cuda_device)
        box_categories = box_categories.to(cuda_device)
        box_tensors = box_tensors.to(cuda_device)
        
        if not isinstance(word2vec_features, list):  # word2vec has dummpy [] holder if no word2vec provided
           word2vec_features = word2vec_features.to(cuda_device)

        video_label = video_label.to(cuda_device)

        # check_gpu('start val')
        # compute output
        with torch.no_grad():
            output = model(global_img_tensors, box_categories, box_tensors, word2vec_features, video_label, is_inference=True)
            output = output.view((-1, len(val_loader.dataset.classes)))
            #loss = criterion(output, video_label.long().cuda())
            loss = criterion(output.cpu(), video_label.cpu())

            acc1, acc5 = accuracy(output.cpu(), video_label.cpu(), topk=(1, 5))
            if args.evaluate:
                logits_matrix.append(output.cpu().data.numpy())
                targets_list.append(video_label.cpu().numpy())

        # check_gpu('before loss update')

        # measure accuracy and record loss
        losses.update(loss.item(), global_img_tensors.size(0))
        acc_top1.update(acc1.item(), global_img_tensors.size(0))
        acc_top5.update(acc5.item(), global_img_tensors.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i + 1 == len(val_loader):
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc1 {acc_top1.val:.1f} ({acc_top1.avg:.1f})\t'
                  'Acc5 {acc_top5.val:.1f} ({acc_top5.avg:.1f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses, acc_top1=acc_top1, acc_top5=acc_top5))

    if args.evaluate:
        logits_matrix = np.concatenate(logits_matrix)
        targets_list = np.concatenate(targets_list)
        save_results(logits_matrix, targets_list, class_to_idx, args)

    if epoch is not None and tb_logger is not None:
        logs = OrderedDict()
        logs['Val_EpochLoss'] = losses.avg
        logs['Val_EpochAcc@1'] = acc_top1.avg
        logs['Val_EpochAcc@5'] = acc_top5.avg
        # how many iterations we have trained
        for key, value in logs.items():
            tb_logger.log_scalar(value, key, epoch + 1)

        tb_logger.flush()

    return losses.avg, acc_top1.avg, acc_top5.avg


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,), return_predict=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred_T = pred.t()
        correct = pred_T.eq(target.view(1, -1).expand_as(pred_T))
   

        acc = []

        for k in topk:
            #correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            acc.append(correct_k.mul_(100.0 / batch_size))

        if not return_predict:
            return acc
        else:
            return acc, pred



if __name__ == '__main__':
    main()