"""
Run evaluation on a trained model to get mAP and class wise AP.

USAGE:
python eval.py --data data_configs/voc.yaml --weights outputs/training/fasterrcnn_convnext_small_voc_15e_noaug/best_model.pth --model fasterrcnn_convnext_small
"""
from datasets import (
    create_valid_dataset, create_valid_loader
)
from models.create_fasterrcnn_model import create_model
from torch_utils import utils
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from pprint import pprint
from tqdm import tqdm
import torchvision.ops as ops

import torch
import argparse
import yaml
import torchvision
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data', 
        default='data_configs/test_image_config.yaml',
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model', 
        default='fasterrcnn_resnet50_fpn',
        help='name of the model'
    )
    parser.add_argument(
        '-mw', '--weights', 
        default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-ims', '--imgsz', 
        default=640, 
        type=int, 
        help='image size to feed to the network'
    )
    parser.add_argument(
        '-w', '--workers', default=4, type=int,
        help='number of workers for data processing/transforms/augmentations'
    )
    parser.add_argument(
        '-b', '--batch', 
        default=8, 
        type=int, 
        help='batch size to load the data'
    )
    parser.add_argument(
        '-d', '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='show class-wise mAP'
    )
    parser.add_argument(
        '-st', '--square-training',
        dest='square_training',
        action='store_true',
        help='Resize images to square shape instead of aspect ratio resizing \
              for single image training. For mosaic training, this resizes \
              single images to square shape first then puts them on a \
              square canvas.'
    )
    args = vars(parser.parse_args())

    # Load the data configurations
    with open(args['data']) as file:
        data_configs = yaml.safe_load(file)

    # Validation settings and constants.
    try: # Use test images if present.
        VALID_DIR_IMAGES = data_configs['TEST_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['TEST_DIR_LABELS']
    except: # Else use the validation images.
        VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
        VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
    NUM_CLASSES = data_configs['NC']
    CLASSES = data_configs['CLASSES']
    NUM_WORKERS = args['workers']
    DEVICE = args['device']
    BATCH_SIZE = args['batch']

    # Model configurations
    IMAGE_SIZE = args['imgsz']

    # Load the pretrained model
    create_model = create_model[args['model']]
    if args['weights'] is None:
        try:
            model, coco_model = create_model(num_classes=NUM_CLASSES, coco_model=True)
        except:
            model = create_model(num_classes=NUM_CLASSES, coco_model=True)
        if coco_model:
            COCO_91_CLASSES = data_configs['COCO_91_CLASSES']
            valid_dataset = create_valid_dataset(
                VALID_DIR_IMAGES, 
                VALID_DIR_LABELS, 
                IMAGE_SIZE, 
                COCO_91_CLASSES, 
                square_training=args['square_training']
            )

    # Load weights.
    if args['weights'] is not None:
        model = create_model(num_classes=NUM_CLASSES, coco_model=False)
        checkpoint = torch.load(args['weights'], map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        valid_dataset = create_valid_dataset(
            VALID_DIR_IMAGES, 
            VALID_DIR_LABELS, 
            IMAGE_SIZE, 
            CLASSES,
            square_training=args['square_training']
        )
    model.to(DEVICE).eval()
    
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)
    
    # Compute True Positives (TP), False Positives (FP), and False Negatives (FN) based on IoU thresholding.
    def compute_tp_fp_fn(preds_dict, true_dict, iou_threshold=0.50):

        pred_boxes = preds_dict['boxes']
        pred_labels = preds_dict['labels']
        pred_scores = preds_dict['scores']
        gt_boxes = true_dict['boxes']
        gt_labels = true_dict['labels']

        if len(pred_boxes) == 0:
            return 0, 0, len(gt_boxes)  # No predictions, all GT are FN

        if len(gt_boxes) == 0:
            return 0, len(pred_boxes), 0  # No GT, all predictions are FP

        # Compute IoUs between predicted and ground truth boxes
        ious = ops.box_iou(pred_boxes, gt_boxes)

        tp = 0
        fp = 0
        fn = 0
        matched = torch.zeros(len(gt_boxes))  # Track matched GT boxes

        for i, (box, label) in enumerate(zip(pred_boxes, pred_labels)):
            max_iou, max_idx = ious[i].max(0)  # Find best-matching GT box
            if max_iou >= iou_threshold and label == gt_labels[max_idx] and matched[max_idx] == 0:
                tp += 1
                matched[max_idx] = 1  # Mark this GT as matched
            else:
                fp += 1

        fn = len(gt_boxes) - matched.sum().item()  # GT boxes not matched

        return tp, fp, fn

    @torch.inference_mode()
    def evaluate(
        model, 
        data_loader, 
        device, 
        out_dir=None,
        classes=None,
        colors=None
    ):
        metric = MeanAveragePrecision(class_metrics=args['verbose'])
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Test:"
        
        # IoU threshold for a prediction to be considered a True Positive
        iou_threshold = 0.50

        target = []
        preds = []
        counter = 0
        total_tp = 0  # Initialize True Positives
        total_fp = 0  # Initialize False Positives
        total_fn = 0  # Initialize False Negatives
        
        for images, targets in tqdm(metric_logger.log_every(data_loader, 100, header), total=len(data_loader)):
            counter += 1
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            with torch.no_grad():
                outputs = model(images)

            #####################################
            for i in range(len(images)):
                true_dict = dict()
                preds_dict = dict()
                true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
                true_dict['labels'] = targets[i]['labels'].detach().cpu()
                
                # Check if outputs[i] contains valid detections
                if 'boxes' in outputs[i] and len(outputs[i]['boxes']) > 0:
                    preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                    preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                    preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
                else:
                    preds_dict['boxes'] = torch.empty((0, 4))  # Empty tensor for boxes
                    preds_dict['scores'] = torch.empty((0,))
                    preds_dict['labels'] = torch.empty((0,), dtype=torch.int64)
                    
                preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
                preds.append(preds_dict)
                target.append(true_dict)
                
                # Compute TP, FP, FN only if there are predictions
                if len(preds_dict['boxes']) > 0:
                    tp, fp, fn = compute_tp_fp_fn(preds_dict, true_dict, iou_threshold)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
            #####################################
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        torch.set_num_threads(n_threads)
        metric.update(preds, target)
        metric_summary = metric.compute()
        
        # Compute final precision and recall
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

        # Add precision & recall to metric_summary dictionary
        metric_summary['precision'] = precision
        metric_summary['recall'] = recall
        
        return metric_summary

    stats = evaluate(
        model, 
        valid_loader, 
        device=DEVICE,
        classes=CLASSES,
    )

    print('\n')
    print("\n===== Evaluation Metrics =====")
    print(f"mAP@50-95: {stats['map']:.3f}")
    print(f"mAP@50: {stats['map_50']:.3f}")

    # Leave a few lines for clarity
    print("\n")

    print(f"Precision: {stats['precision']:.3f}")
    print(f"Recall: {stats['recall']:.3f}")

    print("\n===== Class-wise AP and AR =====")
    pprint(stats)
    if args['verbose']:
        print('\n')
        pprint(f"Classes: {CLASSES}")
        print('\n')
        print('AP / AR per class')
        empty_string = ''
        if len(CLASSES) > 2: 
            num_hyphens = 73
            print('-'*num_hyphens)
            print(f"|    | Class{empty_string:<16}| AP{empty_string:<18}| AR{empty_string:<18}|")
            print('-'*num_hyphens)
            class_counter = 0
            for i in range(0, len(CLASSES)-1, 1):
                class_counter += 1
                print(f"|{class_counter:<3} | {CLASSES[i+1]:<20} | {np.array(stats['map_per_class'][i]):.3f}{empty_string:<15}| {np.array(stats['mar_100_per_class'][i]):.3f}{empty_string:<15}|")
            print('-'*num_hyphens)
            print(f"|Avg{empty_string:<23} | {np.array(stats['map']):.3f}{empty_string:<15}| {np.array(stats['mar_100']):.3f}{empty_string:<15}|")
        else:
            num_hyphens = 62
            print('-'*num_hyphens)
            print(f"|Class{empty_string:<10} | AP{empty_string:<18}| AR{empty_string:<18}|")
            print('-'*num_hyphens)
            print(f"|{CLASSES[1]:<15} | {np.array(stats['map']):.3f}{empty_string:<15}| {np.array(stats['mar_100']):.3f}{empty_string:<15}|")
            print('-'*num_hyphens)
            print(f"|Avg{empty_string:<12} | {np.array(stats['map']):.3f}{empty_string:<15}| {np.array(stats['mar_100']):.3f}{empty_string:<15}|")