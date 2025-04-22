from segment_anything import sam_model_registry
import torch.nn as nn
import torch
import argparse
import os
from utils import FocalDiceloss_IoULoss, generate_point, save_masks
from torch.utils.data import DataLoader
from DataLoader import TestingDataset
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
import logging
import datetime
import cv2
import random
import csv
import json
from DataLoader import TrainingDataset, stack_dict_batched, TrainingDataset_unet
from utils import generate_point_self
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--work_dir", type=str, default="workdir_fold10", help="work dir")
    # parser.add_argument("--work_dir", type=str, default="workdir_aug", help="work dir")
    parser.add_argument("--work_dir", type=str, default="workdir_ex", help="work dir")
    parser.add_argument("--data_path", type=str, default="data/fold_10/", help="train data path")
    # parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    # parser.add_argument("--data_path", type=str, default="data_demo/fold_10/", help="train data path")

    parser.add_argument("--run_name", type=str, default="m3", help="run model name")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    

    parser.add_argument("--metrics", nargs='+', default=['dice', 'dice_1', 'dice_2', 'dice_3'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--model_name", type=str, default="SAM_Med2d", help="SAM_Med2d")
    parser.add_argument("--cls_emb", type=bool, default=False, help="use class embedding")
    parser.add_argument("--fold", type=int, default=0, help="fold number")
    parser.add_argument("--sam_checkpoint", type=str, default="./pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--num_sample", type=int, default=1, help="num of samples") # 2,4,8,16,32
    parser.add_argument("--fusion_ratio", type=float, default=0.1, help="ratio of agent") 
    parser.add_argument("--train", type=str, default='false') 
    parser.add_argument("--point_num", type=int, default=5)

    args = parser.parse_args()
    return args 


def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
        )
    
    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')  #(image_size - ori_h) // 2
        left = torch.div((image_size - ori_w), 2, rounding_mode='trunc') #(image_size - ori_w) // 2
        masks = masks[..., top : ori_h + top, left : ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None 
    return masks, pad


def prompt_and_decoder(args, batched_input, ddp_model, image_embeddings):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = ddp_model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

        low_res_masks, iou_predictions = ddp_model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = ddp_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
    

    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions


def is_not_saved(save_path, mask_name):
    masks_path = os.path.join(save_path, f"{mask_name}")
    if os.path.exists(masks_path):
        return False
    else:
        return True


def main(args):
    print('*'*100)
    for key, value in vars(args).items():
        print(key + ': ' + str(value))
    print('*'*100)

    model = sam_model_registry[args.model_type](args).to(args.device) 

    test_dataset = TestingDataset(data_path=args.data_path, image_size=args.image_size, mode='test', requires_name=True, point_num=args.point_num, return_ori_mask=True, prompt_path=args.prompt_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print('Test data:', len(test_loader))

    test_pbar = tqdm(test_loader)
    l = len(test_loader)

    model.eval()
    test_loss = []
    test_iter_metrics = [0] * len(args.metrics)
    test_metrics = {}
    prompt_dict = {}

    for i, batched_input in enumerate(test_pbar):
        batched_input = to_device(batched_input, args.device)
        ori_labels = batched_input["ori_label"]
        original_size = batched_input["original_size"]
        labels = batched_input["label"]
        img_name = batched_input['name'][0]
        if args.prompt_path is None:
            prompt_dict[img_name] = {
                        "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                        "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                        "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
                        }

        with torch.no_grad():
            image_embeddings = model.image_encoder(batched_input["image"])

        
        save_path = os.path.join(f"{args.work_dir}", args.run_name, f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt")
        batched_input["boxes"] = None
        point_coords, point_labels = [batched_input["point_coords"]], [batched_input["point_labels"]]
    
        for iter in range(args.iter_point):
            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
            if iter != args.iter_point-1:
                batched_input = generate_point(masks, labels, low_res_masks, batched_input, args.point_num)
                batched_input = to_device(batched_input, args.device)
                point_coords.append(batched_input["point_coords"])
                point_labels.append(batched_input["point_labels"])
                batched_input["point_coords"] = torch.concat(point_coords,dim=1)
                batched_input["point_labels"] = torch.concat(point_labels, dim=1)

        points_show = (torch.concat(point_coords, dim=1), torch.concat(point_labels, dim=1))

        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)
        if args.save_pred:
            save_masks(masks, save_path, img_name, args.image_size, original_size, pad, batched_input.get("boxes", None), points_show)


        test_batch_metrics = SegMetrics(masks, ori_labels, args.metrics)
        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]

        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]
  
    test_iter_metrics = [metric / l for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}



def eval_one_epoch(args, model, train_loader):
    train_loader = tqdm(train_loader, ncols=100)
    model.eval()

    valid_iter_metrics = [0] * len(args.metrics)

    valid_metrics_all = {}
    for metric_name in args.metrics:
        valid_metrics_all[metric_name] = []

    for batch, batched_input in enumerate(train_loader):
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)


        labels = batched_input["label"]
        image_input = batched_input['image']
        # labels_edge = batched_input['edge']
        b,c,h,w = labels.shape


        with torch.no_grad():
            image_embeddings = model.image_encoder(image_input).detach()
        list_image_embeds = []
        for img_emb in image_embeddings:
            for _ in range(c):
                list_image_embeds.append(img_emb)
        image_embeddings = torch.stack(list_image_embeds, dim=0)


        point_coords = []
        point_labels = []

        temp_lbl = batched_input['label']
        list_lbls = []
        for i in range(c):
            list_lbls.append(temp_lbl[:,i:i+1,:,:])
        temp_lbls = torch.cat(list_lbls, dim=0)
        batched_input['label'] = temp_lbls

        num_pt_current = 0
        num_pt_sampler = 2

        with torch.no_grad():
            batched_input = generate_point_self(batched_input['label'], None, None, batched_input, 4, status='valid')
            batched_input = to_device(batched_input, args.device)
            point_coords.append(batched_input["point_coords"])
            point_labels.append(batched_input["point_labels"])
            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
            num_pt_current += 4

            while(num_pt_current<args.point_num):
                batched_input = generate_point(masks, batched_input['label'], low_res_masks, batched_input, num_pt_sampler)
                batched_input = to_device(batched_input, args.device)
                point_coords.append(batched_input["point_coords"])
                point_labels.append(batched_input["point_labels"])
                batched_input["point_coords"] = torch.concat(point_coords,dim=1)
                batched_input["point_labels"] = torch.concat(point_labels, dim=1)
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
                num_pt_current += num_pt_sampler
        

        list_res_masks = []
        for i in range(0,b*c,2):
            list_res_masks.append(low_res_masks[i:i+2,0,:,:])
        
        low_res_masks = torch.stack(list_res_masks, dim=0)

        mask_sam = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="nearest",)

        save_mask(batched_input, mask_sam)


        gpu_info = {}
        gpu_info['gpu_name'] = args.device


        valid_batch_metrics = SegMetrics(mask_sam, labels, args.metrics)
        valid_iter_metrics = [valid_iter_metrics[i] + valid_batch_metrics[i] for i in range(len(args.metrics))]

        
        for i in range(len(args.metrics)):
            valid_metrics_all[args.metrics[i]].append(float(valid_batch_metrics[i]))

    return valid_iter_metrics, valid_metrics_all

def save_mask(batched_input, mask):
    save_dir = './sam_vis/'
    os.makedirs(save_dir, exist_ok=True)
    mask_show = F.sigmoid(mask.detach()).permute(0,2,3,1).cpu().numpy()[0]
    mask_show[mask_show>0.5] = 1.0
    mask_show[mask_show<1] = 0.0
    
    mask_show = np.concatenate([mask_show, np.zeros_like(mask_show[:,:,0:1])], axis=-1)
    mask_show = mask_show*255
    mask_show = mask_show.copy().astype(np.uint8)



    ## 3*n*2;3*n
    pts = batched_input['point_coords'].view(-1,2).cpu().numpy().tolist()
    pts_lbl = batched_input["point_labels"].view(-1).cpu().numpy().tolist()
    

    for pt, pt_lbl in zip(pts, pts_lbl):
        x,y = pt
        if pt_lbl==1:
            color = (255,255,255)
            cv2.drawMarker(mask_show, (int(x), int(y)), color, markerType=cv2.MARKER_STAR , markerSize=5, thickness=1)
        else:
            color = (255,255,255)
            cv2.drawMarker(mask_show, (int(x), int(y)), color, markerType=cv2.MARKER_CROSS , markerSize=5, thickness=1)
    
    temp_img_dir = save_dir + str(batched_input['name'][0]) + '.jpg'
    # cv2.imwrite(temp_img_dir, mask_show)
    plt.imsave(temp_img_dir, mask_show)




def eval(args):
    model = sam_model_registry[args.model_type](args).to(args.device) 
    test_dataset = TrainingDataset_unet(args.data_path, image_size=args.image_size, mode='test', requires_name = False, fold=args.fold, num_sample=-1, num_fold=10)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=0)
    print('*******test data:', len(test_dataset)) 

    temp_save_dir = os.path.join(args.work_dir, "models", str(args.num_sample), args.model_name)
    os.makedirs(temp_save_dir, exist_ok=True)
    test_iter_metrics, test_metrics_all = eval_one_epoch(args, model, test_loader)
    test_iter_metrics = [metric / len(test_loader) for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}
    temp_metic = np.mean([float(test_metrics['dice_1']),float(test_metrics['dice_2'])])

    test_metrics['dice'] = str('{:.4f}'.format(temp_metic))
    print(test_metrics)

    temp_save_dir = os.path.join(args.work_dir, "results", str(args.num_sample), args.model_name)
    os.makedirs(temp_save_dir, exist_ok=True)
    save_path_json = os.path.join(temp_save_dir, f"temp_{args.model_name}_fold{str(args.fold)}.json")
    with  open(save_path_json, mode='w') as f:
        json.dump(test_metrics_all, f)

    return test_metrics



if __name__ == '__main__':
    args = parse_args()
    args.metrics = ['dice_1','dice_2', 'hd_1', 'hd_2',  
                        'sen_1', 'sen_2', 'spec_1', 'spec_2', 
                        'auc_1', 'auc_2', 'aupr_1', 'aupr_2',
                        ]
    args.batch_size = 1
    eval(args)