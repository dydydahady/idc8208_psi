from opts import get_opts
from datetime import datetime
import os
import json
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data.prepare_data import get_dataloader
from database.create_database import create_database
from models.build_model import build_model
from train import train_intent
from test import validate_intent, test_intent, predict_intent
from utils.log import RecordResults
from utils.evaluate_results import evaluate_intent
from utils.get_test_intent_gt import get_intent_gt
from data.custom_dataset import VideoDataset


def main(args):
    writer = SummaryWriter(log_dir=args.checkpoint_path)
    recorder = RecordResults(args)
    print("Path to video splits:", args.video_splits)

    ''' 1. Load database '''
    if not os.path.exists(os.path.join(args.database_path, args.database_file)):
        create_database(args)
    else:
        print("Database exists!")

    train_loader, val_loader = get_dataloader(args)

    ''' 2. Create models '''
    model, optimizer, scheduler = build_model(args)
    model = nn.DataParallel(model)

    ''' 3. Class Weights for Imbalanced Dataset '''
    # Assuming you have loaded your dataset, for example:
    train_dataset = VideoDataset(data=train_loader.dataset.data, args=args)

    # Retrieve class weights from dataset
    weight_cross = train_dataset.class_weight_cross
    weight_not_cross = train_dataset.class_weight_not_cross

    # Use these weights in setting up BCEWithLogitsLoss
    pos_weight = torch.tensor([weight_cross]).to(args.device)
    criterions = {
        'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none').to(args.device),
        'MSELoss': torch.nn.MSELoss(reduction='none').to(args.device),
    }

    ''' 4. Train '''
    train_intent(model, optimizer, scheduler, train_loader, val_loader, args, recorder, writer, criterions)

    val_gt_file = '../dataset/val_intent_gt.json'
    if not os.path.exists(val_gt_file):
        get_intent_gt(val_loader, val_gt_file, args)
    predict_intent(model, val_loader, args, dset='val')
    

    ''' 5. Evaluate '''
    evaluate_intent(val_gt_file, args.checkpoint_path + '/results/val_intent_pred.json', args, writer)
    

    # Closing TensorBoard writer
    writer.close()


if __name__ == '__main__':
    args = get_opts()
    args.dataset_root_path = '/home/dydy/proj_idc8208/dataset/'
    
    # Dataset
    args.dataset = 'PSI2.0'
    if args.dataset == 'PSI2.0':
        args.video_splits = '/home/dydy/proj_idc8208/dataset/PSI2.0_TrainVal/splits/PSI2_split.json'
    elif args.dataset == 'PSI1.0':
        args.video_splits = os.path.join(args.dataset_root_path, 'PSI1.0/splits/PSI1_split.json')
    else:
        raise Exception("Unknown dataset name!")

    # Task
    args.task_name = 'ped_intent'

    if args.task_name == 'ped_intent':
        args.database_file = 'intent_database_train.pkl'
        args.intent_model = True

    # intent prediction
    args.intent_num = 2  # 3 for 'major' vote; 2 for mean intent
    args.intent_type = 'mean' # >= 0.5 --> 1 (cross); < 0.5 --> 0 (not cross)
    args.intent_loss = ['bce']
    args.intent_disagreement = 1  # -1: not use disagreement 1: use disagreement to reweigh samples
    args.intent_positive_weight = 0.5  # Reweigh BCE loss of 0/1, 0.5 = count(-1) / count(1)

    # trajectory
    if args.task_name == 'ped_traj':
        args.database_file = 'traj_database_train.pkl'
        args.intent_model = False
        args.traj_model = True
        args.traj_loss = ['bbox_l1']

    args.seq_overlap_rate = 0.9
    args.test_seq_overlap_rate = 1
    args.observe_length = 15
    if args.task_name == 'ped_intent':
        args.predict_length = 1
    elif args.task_name == 'ped_traj':
        args.predict_length = 45

    args.max_track_size = args.observe_length + args.predict_length
    args.crop_mode = 'enlarge'
    args.normalize_bbox = None

    # Model
    args.model_name = 'transformer_int_bbox'
    args.load_image = False
    if args.load_image:
        args.backbone = 'resnet'
        args.freeze_backbone = False
    else:
        args.backbone = None
        args.freeze_backbone = False

    # Train
    args.epochs = 50
    args.batch_size = 128
    args.lr = 1e-5
    args.loss_weights = {
        'loss_intent': 1.0,
        'loss_traj': 0.0,
        'loss_driving': 0.0
    }
    args.val_freq = 1
    args.test_freq = 1
    args.print_freq = 10

    # Record
    now = datetime.now()
    time_folder = now.strftime('%Y%m%d%H%M%S')
    args.checkpoint_path = os.path.join(args.checkpoint_path, args.task_name, args.dataset, args.model_name, time_folder)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    with open(os.path.join(args.checkpoint_path, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    result_path = os.path.join(args.checkpoint_path, 'results')
    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    main(args)
