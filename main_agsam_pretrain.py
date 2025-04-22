from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn
import torch
import argparse
import os
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, stack_dict_batched, TrainingDataset_unet
from utils import FocalDiceloss_IoULoss, get_logger, DiceLoss, FocalLoss, FocalDiceloss
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
import datetime
from torch.nn import functional as F
# from apex import amp
import random
import json

from model_s import NestedUNet, PSPNet, FastSCNN, TGAPolypSeg
# from torchvision.models.segmentation import  deeplabv3_resnet50
from segformer_pytorch.segformer_pytorch import Segformer
from model_fcn import fcn_resnet50_my
from model_deeplab import deeplabv3_resnet50_my
import seaborn as sns
import matplotlib.pyplot as plt
# import random
seed = 2023
random.seed(seed)
# import numpy as np
np.random.seed(seed)
# import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
from fvcore.nn import FlopCountAnalysis, parameter_count_table

resume_dir = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir_fold10", help="work dir")
    parser.add_argument("--data_path", type=str, default="data/fold_10/", help="train data path")
    # parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    # parser.add_argument("--data_path", type=str, default="data/fold_10/", help="train data path")

    parser.add_argument("--run_name", type=str, default="m3", help="run model name")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    

    parser.add_argument("--metrics", nargs='+', default=['dice', 'dice_1', 'dice_2',], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--model_name", type=str, default="DeepLabv3", help="model names [FCN, DeepLabV3++, PSPNet, Fast-SCNN, SegFormer, Mask2Former, TGANet, Unet]")
    parser.add_argument("--cls_emb", type=bool, default=False, help="use class embedding")
    parser.add_argument("--fold", type=int, default=0, help="fold number")
    parser.add_argument("--sam_checkpoint", type=str, default="./pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--num_sample", type=int, default=1, help="num of samples") # 2,4,8,16,32
    parser.add_argument("--fusion_ratio", type=float, default=0.5, help="ratio of agent") 
    parser.add_argument("--train", type=str, default='true') 

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


def prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    if decoder_iter:
        with torch.no_grad():
            if args.cls_emb:
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=points,
                    boxes=batched_input.get("boxes", None),
                    masks=batched_input.get("mask_inputs", None),
                    clss=batched_input['cls'],
                )
            else:
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=points,
                    boxes=batched_input.get("boxes", None),
                    masks=batched_input.get("mask_inputs", None),
                    clss=None,
                )

    else:
        if args.cls_emb:
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
                clss=batched_input['cls'],
            )
        else:
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
                clss=None,
            )

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )
  
    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    # masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="nearest", align_corners=False,)
    
    # masks = model.mask(low_res_masks)
    return masks, low_res_masks, iou_predictions


# def train_one_epoch(args, model, optimizer, train_loader, criterion, model_agent, optimizer_agent, optimizer_agent_prompt):
#     train_loader = tqdm(train_loader, ncols=100)
#     train_losses = []
#     train_iter_metrics = [0] * len(args.metrics)
#     model.train()
#     model_agent.train()
#     for batch, batched_input in enumerate(train_loader):
#         batched_input = stack_dict_batched(batched_input)
#         batched_input = to_device(batched_input, args.device)
        
        
#         labels = batched_input["label"]
#         image_input = batched_input['image']
#         # labels_edge = batched_input['edge']
        
        
#         ## step 2 model agent for prompt
#         for n, value in model_agent.named_parameters():
#             if "prompt" in n or 'mask_up' in n:
#                 value.requires_grad = True
#             else:
#                 value.requires_grad = False
#         with torch.no_grad():
#             image_embeddings = model.image_encoder(image_input).detach()
#             image_embeddings_ori = image_embeddings.clone()
#         out = model_agent(image_input, image_embeddings)
#         if args.model_name == 'Unet' or args.model_name == 'Unet_sam' or args.model_name == 'Unet_sam_joint':
#             masks = out['masks']
#         elif args.model_name == 'FCN' or args.model_name == 'DeepLabv3' or args.model_name == 'PSPNet' or args.model_name == 'Fast-SCNN' or args.model_name == 'TGANet':
#             mask = out['out']
#             masks = [mask]

        
#         b,c,h,w = masks[-1].shape
#         sparse_embeddings = out['points_prompt'].view(b*2, -1, 256) 
#         dense_embeddings = out['masks_prompt'].view(b*2, 256, h//16, w//16)


#         list_image_embeds = []
#         for img_emb in image_embeddings:
#             for _ in range(c):
#                 list_image_embeds.append(img_emb)
#         image_embeddings = torch.stack(list_image_embeds, dim=0)


#         # with torch.no_grad():
#         # sparse_embeddings  (b*3)*n*256
#         # dense_embeddings   (b*3)*256*16*16
#         low_res_masks, iou_predictions = model.mask_decoder(
#                         image_embeddings = image_embeddings,
#                         image_pe = model.prompt_encoder.get_dense_pe(),
#                         sparse_prompt_embeddings=sparse_embeddings,
#                         dense_prompt_embeddings=dense_embeddings,
#                         multimask_output=False,
#                         )

#         list_res_masks = []
#         for i in range(0,b*c,2):
#             list_res_masks.append(low_res_masks[i:i+2,0,:,:])
        
#         low_res_masks = torch.stack(list_res_masks, dim=0)
#         mask_sam = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="nearest")
#         # mask_sam = model_agent.mask_up(low_res_masks)

        
#         list_ratios = [args.fusion_ratio]
#         loss_sam = 0
#         for m in masks:
#             # for temp_r in list_ratios:
#             mask_fusion = F.sigmoid(mask_sam)*(1-args.fusion_ratio) + F.sigmoid(m)*args.fusion_ratio
#             # mask_fusion = F.sigmoid(mask_sam*(1-args.fusion_ratio) + m*args.fusion_ratio)
#             loss_sam += criterion(mask_fusion, labels)


#         # mask_fusion = F.sigmoid(mask_sam)*(1-args.fusion_ratio) + F.sigmoid(masks[-1])*args.fusion_ratio
#         # mask_fusion = F.sigmoid(mask_sam*(1-args.fusion_ratio) + masks[-1]*args.fusion_ratio)
#         # mask_fusion = mask_sam*(1-args.fusion_ratio) + masks[-1]*args.fusion_ratio 
#         # loss_sam = criterion(mask_fusion, labels)
        
        
#         loss_sam.backward(retain_graph=True)
        


#         optimizer_agent_prompt.step()
#         optimizer_agent_prompt.zero_grad()
#         train_losses.append(loss_sam.item())

        
#         # mask_fusion = F.sigmoid(masks[-1])
#         # loss_sam = loss_mask

#         mask_fusion = F.sigmoid(mask_sam)*(1-args.fusion_ratio) + F.sigmoid(masks[-1])*args.fusion_ratio
#         # mask_fusion = F.sigmoid(mask_sam*(1-args.fusion_ratio) + masks[-1]*args.fusion_ratio)

        
        

#         gpu_info = {}
#         gpu_info['gpu_name'] = args.device
#         # train_loader.set_postfix(loss_sam=loss_sam.item())
#         temp_loss = str('{:.5f}'.format(loss_sam.item()))
#         train_loader.set_postfix(train_loss=temp_loss)

#         # train_batch_metrics = SegMetrics(mask_fusion, labels, args.metrics)
#         # train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]

#         # train_batch_metrics = SegMetrics(masks[-1], labels, args.metrics)
#         # train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]
#         # train_batch_metrics = SegMetrics(mask_sam, labels, args.metrics)
#         # train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]
        

#     return train_losses, train_iter_metrics


# def eval_one_epoch(args, model, train_loader, model_agent):
#     train_loader = tqdm(train_loader, ncols=100)
#     model.eval()
#     model_agent.eval()
#     valid_iter_metrics = [0] * len(args.metrics)

#     valid_metrics_all = {}
#     for metric_name in args.metrics:
#         valid_metrics_all[metric_name] = []

#     for batch, batched_input in enumerate(train_loader):
#         batched_input = stack_dict_batched(batched_input)
#         batched_input = to_device(batched_input, args.device)
        
        
#         labels = batched_input["label"]
#         image_input = batched_input['image']
#         # labels_edge = batched_input['edge']


#         with torch.no_grad():
#             image_embeddings = model.image_encoder(image_input).detach()
#             out = model_agent(image_input, image_embeddings)
#         if args.model_name == 'Unet' or args.model_name == 'Unet_sam' or args.model_name == 'Unet_sam_joint':
#             masks = out['masks']
#         elif args.model_name == 'FCN' or args.model_name == 'DeepLabv3' or args.model_name == 'PSPNet' or args.model_name == 'Fast-SCNN' or args.model_name == 'TGANet':
#             mask = out['out']
#             masks = [mask]

        
#         b,c,h,w = masks[-1].shape
#         sparse_embeddings = out['points_prompt'].view(b*2, -1, 256) 
#         dense_embeddings = out['masks_prompt'].view(b*2, 256, h//16, w//16)
        

#         list_image_embeds = []
#         for img_emb in image_embeddings:
#             for _ in range(c):
#                 list_image_embeds.append(img_emb)
#         image_embeddings = torch.stack(list_image_embeds, dim=0)
        
#         with torch.no_grad():

            
#             low_res_masks, iou_predictions = model.mask_decoder(
#             image_embeddings = image_embeddings,
#             image_pe = model.prompt_encoder.get_dense_pe(),
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings,
#             multimask_output=False,
#             )

#             list_res_masks = []
#             for i in range(0,b*c,2):
#                 list_res_masks.append(low_res_masks[i:i+2,0,:,:])
            
#             low_res_masks = torch.stack(list_res_masks, dim=0)

#             mask_sam = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="nearest",)
#             # mask_sam = model_agent.mask_up(low_res_masks)



        

        
        
#         # mask_fusion = F.sigmoid(mask_sam)*(1-args.fusion_ratio) + F.sigmoid(masks[-1])*args.fusion_ratio
#         # mask_fusion = F.sigmoid(mask_sam*(1-args.fusion_ratio) + masks[-1]*args.fusion_ratio)
#         # mask_fusion = mask_sam*(1-args.fusion_ratio) + masks[-1]*args.fusion_ratio 
        

#         mask_fusion = F.sigmoid(masks[-1])
        
#         gpu_info = {}
#         gpu_info['gpu_name'] = args.device


#         valid_batch_metrics = SegMetrics(mask_fusion, labels, args.metrics)
#         valid_iter_metrics = [valid_iter_metrics[i] + valid_batch_metrics[i] for i in range(len(args.metrics))]



#         # valid_batch_metrics = SegMetrics(masks[-1], labels, args.metrics)
#         # valid_iter_metrics = [valid_iter_metrics[i] + valid_batch_metrics[i] for i in range(len(args.metrics))]
#         # valid_batch_metrics = SegMetrics(mask_sam, labels, args.metrics)
#         # valid_iter_metrics = [valid_iter_metrics[i] + valid_batch_metrics[i] for i in range(len(args.metrics))]
        
#         for i in range(len(args.metrics)):
#             valid_metrics_all[args.metrics[i]].append(float(valid_batch_metrics[i]))


#     model.train()
#     model_agent.train()
#     return valid_iter_metrics, valid_metrics_all




def train_one_epoch(args, model, optimizer, train_loader, criterion, model_sam):
    train_loader = tqdm(train_loader, ncols=100)
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)
    model_sam.eval()
    for batch, batched_input in enumerate(train_loader):
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        
        
        labels = batched_input["label"]
        image_input = batched_input['image']
        # labels_edge = batched_input['edge']
        for n, value in model_sam.named_parameters():
            if "prompt" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False
        with torch.no_grad():
            image_embeddings = model_sam.image_encoder(image_input).detach()


        out = model(image_input, image_embeddings)

        if args.model_name == 'Unet':
            masks = out['masks']
        elif args.model_name == 'FCN' or 'DeepLabv3' or 'PSPNet' or 'Fast-SCNN' or 'TGANet':
            mask = out['out']
            masks = [mask]

        b,c,h,w = masks[-1].shape
        sparse_embeddings = out['points_prompt'].view(b*2, -1, 256) 
        dense_embeddings = out['masks_prompt'].view(b*2, 256, h//16, w//16)

        list_image_embeds = []
        for img_emb in image_embeddings:
            for _ in range(c):
                list_image_embeds.append(img_emb)
        image_embeddings = torch.stack(list_image_embeds, dim=0)

        low_res_masks, iou_predictions = model_sam.mask_decoder(
                        image_embeddings = image_embeddings,
                        image_pe = model_sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        )
        list_res_masks = []
        for i in range(0,b*c,2):
            list_res_masks.append(low_res_masks[i:i+2,0,:,:])
        
        low_res_masks = torch.stack(list_res_masks, dim=0)
        mask_sam = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="nearest")



        
        loss_sam = 0
        for m in masks:
            # for temp_r in list_ratios:
            mask_fusion = F.sigmoid(mask_sam)*(1-args.fusion_ratio) + F.sigmoid(m)*args.fusion_ratio
            # mask_fusion = F.sigmoid(mask_sam*(1-args.fusion_ratio) + m*args.fusion_ratio)
            loss_sam += criterion(mask_fusion, labels)

        loss = loss_sam
        loss.backward(retain_graph=False)

        optimizer.step()
        optimizer.zero_grad()
        train_losses.append(loss.item())

        mask_fusion = F.sigmoid(mask_sam)*(1-args.fusion_ratio) + F.sigmoid(masks[-1])*args.fusion_ratio

        gpu_info = {}
        gpu_info['gpu_name'] = args.device 
        temp_loss = str('{:.5f}'.format(loss.item()))
        train_loader.set_postfix(train_loss=temp_loss)

        train_batch_metrics = SegMetrics(F.sigmoid(masks[-1]), labels, args.metrics)
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]

    return train_losses, train_iter_metrics


def eval_one_epoch(args, model, train_loader, model_sam):
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


        with torch.no_grad():
            image_embeddings = model_sam.image_encoder(image_input).detach()
            out = model(image_input, image_embeddings)
        if args.model_name == 'Unet':
            masks = out['masks']
        elif args.model_name == 'FCN' or 'DeepLabV3' or 'PSPNet' or 'Fast-SCNN' or 'TGANet':
            mask = out['out']
            masks = [mask]
        

        b,c,h,w = masks[-1].shape
        sparse_embeddings = out['points_prompt'].view(b*2, -1, 256) 
        dense_embeddings = out['masks_prompt'].view(b*2, 256, h//16, w//16)
        

        list_image_embeds = []
        for img_emb in image_embeddings:
            for _ in range(c):
                list_image_embeds.append(img_emb)
        image_embeddings = torch.stack(list_image_embeds, dim=0)
        with torch.no_grad():
            low_res_masks, iou_predictions = model_sam.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = model_sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            )

            list_res_masks = []
            for i in range(0,b*c,2):
                list_res_masks.append(low_res_masks[i:i+2,0,:,:])
            
            low_res_masks = torch.stack(list_res_masks, dim=0)

            mask_sam = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="nearest",)

        mask_fusion = F.sigmoid(masks[-1])
        


        gpu_info = {}
        gpu_info['gpu_name'] = args.device 


        valid_batch_metrics = SegMetrics(mask_fusion, labels, args.metrics)
        valid_iter_metrics = [valid_iter_metrics[i] + valid_batch_metrics[i] for i in range(len(args.metrics))]
        for i in range(len(args.metrics)):
            valid_metrics_all[args.metrics[i]].append(float(valid_batch_metrics[i]))
    
    model.train()
    return valid_iter_metrics, valid_metrics_all






import itertools

def train(args):

    temp_save_dir = os.path.join(args.work_dir, "models", str(args.num_sample), args.model_name+'_m1')
    os.makedirs(temp_save_dir, exist_ok=True)
    save_path = os.path.join(temp_save_dir, f"temp_{args.model_name+'_m1'}_fold{str(args.fold)}.pth")
    args.sam_checkpoint = save_path
    ck = torch.load(save_path)['model']
    



    model = sam_model_registry[args.model_type](args).to(args.device) 
    optimizer = optim.AdamW(model.mask_decoder.parameters(), lr=args.lr, weight_decay=5e-4)
    # optimizer = optim.SGD(model.mask_decoder.parameters(), lr=args.lr*0.1, weight_decay=5e-4, momentum=0.9)
    criterion = DiceLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs*1//2], gamma = 0.1)


    if args.model_name == 'Unet':
        model_agent = NestedUNet(2, 3, True, True).to(args.device)
    elif args.model_name == 'FCN':
        model_agent = fcn_resnet50_my(num_classes=2, model_type='m3').to(args.device)
    elif args.model_name == 'DeepLabv3':
        model_agent = deeplabv3_resnet50_my(num_classes=2, model_type='m3').to(args.device)
    elif args.model_name == 'PSPNet':
        model_agent = PSPNet(n_classes=2).to(args.device)
    elif args.model_name == 'Fast-SCNN':
        model_agent = FastSCNN(num_classes=2).to(args.device)
    elif args.model_name == 'SegFormer':
        model_agent = Segformer().to(args.device)
    elif args.model_name == 'TGANet':
        model_agent = TGAPolypSeg(num_classes=2).to(args.device)

    model_agent.load_state_dict(ck, strict=False)
    
    optimizer_agent = optim.AdamW(model_agent.parameters(), lr=args.lr, weight_decay=5e-4)
    # optimizer_agent_prompt = optim.AdamW(itertools.chain(model_agent.mask_prompt.parameters(),model_agent.point_prompt.parameters(), model_agent.mask_up.parameters()), lr=args.lr, weight_decay=5e-4)


    optimizer_agent_prompt = optim.AdamW(model_agent.parameters(), lr=args.lr, weight_decay=5e-4)



    # optimizer_agent = optim.SGD(model_agent.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    # optimizer_agent_prompt = optim.SGD(itertools.chain(model_agent.mask_prompt.parameters(),model_agent.point_prompt.parameters()), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    criterion_agent = DiceLoss()
    scheduler_agent = torch.optim.lr_scheduler.MultiStepLR(optimizer_agent, milestones=[args.epochs//2], gamma = 0.1)
    scheduler_agent_prompt = torch.optim.lr_scheduler.MultiStepLR(optimizer_agent_prompt, milestones=[args.epochs//2], gamma = 0.1)




    train_dataset = TrainingDataset_unet(args.data_path, image_size=args.image_size, mode='train', requires_name = False, fold=args.fold, num_sample=args.num_sample, num_fold=10)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    print('*******Train data:', len(train_dataset))   

    valid_dataset = TrainingDataset_unet(args.data_path, image_size=args.image_size, mode='test', requires_name = False, fold=args.fold, num_sample=-1, num_fold=10)
    valid_loader = DataLoader(valid_dataset, batch_size = 16, shuffle=True, num_workers=0)
    print('*******Valid data:', len(valid_dataset))   

    test_dataset = TrainingDataset_unet(args.data_path, image_size=args.image_size, mode='test', requires_name = False, fold=args.fold, num_sample=-1, num_fold=10)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle=True, num_workers=0)
    print('*******test data:', len(test_dataset)) 

    # loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))
    # l = len(valid_loader)
    best_metric = 0
    list_valid = []
    list_test = []

    for epoch in range(0, args.epochs):
        model.train()
        train_metrics = {}
        start = time.time()

        # _, train_iter_metrics = train_one_epoch(args, model, optimizer, train_loader, criterion, model_agent, optimizer_agent, optimizer_agent_prompt)
        _, train_iter_metrics = train_one_epoch(args, model_agent, optimizer_agent_prompt, train_loader, criterion, model)

        # train_iter_metrics,_ = eval_one_epoch(args, model, train_loader, model_agent)
        train_iter_metrics = [metric / len(train_loader) for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}


        
        # scheduler.step()
        # scheduler_agent.step()
        scheduler_agent_prompt.step()

        valid_metrics = {}
        valid_iter_metrics,_ = eval_one_epoch(args, model_agent, valid_loader, model)
        valid_iter_metrics = [metric / len(valid_loader) for metric in valid_iter_metrics]
        valid_metrics = {args.metrics[i]: '{:.4f}'.format(valid_iter_metrics[i]) for i in range(len(valid_iter_metrics))}
        temp_metic = np.mean([float(valid_metrics['dice_1']),float(valid_metrics['dice_2'])])
        valid_metrics['dice'] = str('{:.4f}'.format(temp_metic))
        lr = scheduler_agent_prompt.get_last_lr()[0]
        print(f"epoch: {epoch + 1}, lr: {lr}, train metrics: {train_metrics}")
        print(f"epoch: {epoch + 1}, lr: {lr}, valid metrics: {valid_metrics}")
        # list_valid.append(float(valid_metrics['dice']))


        if temp_metic >= best_metric and float(train_metrics['dice'])>0.7:
            temp_save_dir = os.path.join(args.work_dir, "models", str(args.num_sample), args.model_name+'_m3')
            os.makedirs(temp_save_dir, exist_ok=True)
            save_path = os.path.join(temp_save_dir, f"temp_{args.model_name+'_m3'}_fold{str(args.fold)}.pth")
            state = {'model': model.float().state_dict(), 
                    #  'optimizer': optimizer, 
                     'model_agent': model_agent.state_dict()}
            torch.save(state, save_path)
            best_metric = temp_metic

        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))

    temp_save_dir = os.path.join(args.work_dir, "models", str(args.num_sample), args.model_name+'_m3')
    save_path = os.path.join(temp_save_dir, f"temp_{args.model_name+'_m3'}_fold{str(args.fold)}.pth")
    args.sam_checkpoint = save_path
    ck = torch.load(save_path)['model_agent']
    model_agent.load_state_dict(ck, strict=False)

    model_sam = sam_model_registry[args.model_type](args).to(args.device) 
    test_iter_metrics,_ = eval_one_epoch(args, model_sam, test_loader, model_agent)
    test_iter_metrics = [metric / len(test_loader) for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}


    temp_metic = np.mean([float(test_metrics['dice_1']),float(test_metrics['dice_2'])])
    test_metrics['dice'] = str('{:.4f}'.format(temp_metic))
    print(test_metrics)
    return test_metrics

        

    # sns.lineplot(list_valid)
    # sns.lineplot(list_test)
    # plt.show()



class model_com(nn.Module):
    def __init__(self, model_agent, model_sam, args):
        super().__init__()
        self.model_agent = model_agent
        self.model = model_sam
        self.args = args

    def forward(self, input):
        args = self.args
        with torch.no_grad():
            image_embeddings = self.model.image_encoder(input).detach()
            out = self.model_agent(input, image_embeddings)
        if args.model_name == 'Unet' or args.model_name == 'Unet_sam' or args.model_name == 'Unet_sam_joint':
            masks = out['masks']
        elif args.model_name == 'FCN' or args.model_name == 'DeepLabv3' or args.model_name == 'PSPNet' or args.model_name == 'Fast-SCNN' or args.model_name == 'TGANet':
            mask = out['out']
            masks = [mask]

        
        b,c,h,w = masks[-1].shape
        sparse_embeddings = out['points_prompt'].view(b*2, -1, 256) 
        dense_embeddings = out['masks_prompt'].view(b*2, 256, h//16, w//16)
        

        list_image_embeds = []
        for img_emb in image_embeddings:
            for _ in range(c):
                list_image_embeds.append(img_emb)
        image_embeddings = torch.stack(list_image_embeds, dim=0)
        
        with torch.no_grad():

            
            low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            )

            list_res_masks = []
            for i in range(0,b*c,3):
                list_res_masks.append(low_res_masks[i:i+2,0,:,:])
            
            low_res_masks = torch.stack(list_res_masks, dim=0)

            mask_sam = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
        
        return mask_sam



def eval(args):

    if args.model_name == 'Unet':
        model_agent = NestedUNet(2, 3, True, True).to(args.device)
    elif args.model_name == 'FCN':
        model_agent = fcn_resnet50_my(num_classes=2, model_type='m3').to(args.device)
    elif args.model_name == 'DeepLabv3':
        model_agent = deeplabv3_resnet50_my(num_classes=2, model_type='m3').to(args.device)
    elif args.model_name == 'PSPNet':
        model_agent = PSPNet(n_classes=2).to(args.device)
    elif args.model_name == 'Fast-SCNN':
        model_agent = FastSCNN(num_classes=2).to(args.device)
    elif args.model_name == 'SegFormer':
        model_agent = Segformer().to(args.device)
    elif args.model_name == 'TGANet':
        model_agent = TGAPolypSeg(num_classes=2).to(args.device)
    

    test_dataset = TrainingDataset_unet(args.data_path, image_size=args.image_size, mode='test', requires_name = False, fold=args.fold, num_sample=-1, num_fold=10)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)
    print('*******test data:', len(test_dataset)) 

    model_sam = sam_model_registry[args.model_type](args).to(args.device) 

    ##analysis module parameters FLops
    temp_model = model_com(model_agent, model_sam, args)
    temp_input = (torch.randn(1,3,256,256).to(args.device))
    flops = FlopCountAnalysis(temp_model, temp_input)
    print('FLOPs:{:.2f}*e10'.format(0.5*flops.total()/1e10))
    print(parameter_count_table(temp_model))
    # return 0



    temp_save_dir = os.path.join(args.work_dir, "models", str(args.num_sample), args.model_name+'_m3')
    os.makedirs(temp_save_dir, exist_ok=True)
    save_path = os.path.join(temp_save_dir, f"temp_{args.model_name+'_m3'}_fold{str(args.fold)}.pth")
    args.sam_checkpoint = save_path
    ck = torch.load(save_path)['model_agent']
    model_agent.load_state_dict(ck, strict=True)

    
    # test_iter_metrics, test_metrics_all = eval_one_epoch(args, model_sam, test_loader, model_agent)
    test_iter_metrics, test_metrics_all = eval_one_epoch(args, model_agent, test_loader, model_sam)

    test_iter_metrics = [metric / len(test_loader) for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}
    temp_metic = np.mean([float(test_metrics['dice_1']),float(test_metrics['dice_2'])])
    test_metrics['dice'] = str('{:.4f}'.format(temp_metic))

    temp_metic = np.mean([float(test_metrics['hd_1']),float(test_metrics['hd_2'])])
    test_metrics['hd'] = str('{:.4f}'.format(temp_metic))
    print(test_metrics)

    temp_save_dir = os.path.join(args.work_dir, "results", str(args.num_sample), args.model_name+'_m3')
    os.makedirs(temp_save_dir, exist_ok=True)
    save_path_json = os.path.join(temp_save_dir, f"temp_{args.model_name+'_m3'}_fold{str(args.fold)}.json")
    with  open(save_path_json, mode='w') as f:
        json.dump(test_metrics_all, f)
    return test_metrics

        

    # sns.lineplot(list_valid)
    # sns.lineplot(list_test)
    # plt.show()



if __name__ == '__main__':
    args = parse_args()
    args.fold = 0
    if args.train=='true':
        train(args)
    else:
        args.metrics = ['dice_1','dice_2', 'hd_1', 'hd_2', 
                        'sen_1', 'sen_2', 'spec_1', 'spec_2', 
                        'auc_1', 'auc_2',  'aupr_1', 'aupr_2', 
                        ]
        eval(args)
    for info in args.__dict__:
        print(info+':'+str(args.__dict__[info]))

    # list_metric = []
    # for i in range(10):
    #     args = parse_args()
    #     args.fold = i
    #     metric = train(args)

    #     list_metric.append(metric)
    
    
    # print('finish')
    # for met in list_metric:
    #     print(met)

    
