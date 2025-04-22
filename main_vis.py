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

import cv2
from models_vm.vmunet import VMUNet

resume_dir = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir_fold10", help="work dir")
    parser.add_argument("--data_path", type=str, default="data/fold_10/", help="train data path")
    # parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    # parser.add_argument("--data_path", type=str, default="data/fold_10/", help="train data path")

    parser.add_argument("--run_name", type=str, default="m3", help="run model name")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    

    parser.add_argument("--metrics", nargs='+', default=['dice', 'dice_1', 'dice_2', 'dice_3'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--model_name", type=str, default="FCN", help="model names [FCN, DeepLabV3++, PSPNet, Fast-SCNN, SegFormer, TGANet, Unet]")
    parser.add_argument("--cls_emb", type=bool, default=False, help="use class embedding")
    parser.add_argument("--fold", type=int, default=0, help="fold number")
    parser.add_argument("--sam_checkpoint", type=str, default="./pretrain_model/sam-med2d_b.pth", help="sam checkpoint")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    parser.add_argument("--num_sample", type=int, default=1, help="num of samples") # 2,4,8,16,32
    parser.add_argument("--fusion_ratio", type=float, default=0.9, help="ratio of agent") 
    parser.add_argument("--train", type=str, default='false') 

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



def vis(args):
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



    temp_save_dir = os.path.join(args.work_dir, "models", str(args.num_sample), args.model_name+'_m3')
    os.makedirs(temp_save_dir, exist_ok=True)
    save_path = os.path.join(temp_save_dir, f"temp_{args.model_name+'_m3'}_fold{str(args.fold)}.pth")
    args.sam_checkpoint = save_path
    ck = torch.load(save_path)['model']
    model_agent.load_state_dict(ck)

    model_sam = sam_model_registry[args.model_type](args).to(args.device)

    
    grad_block = []
    fmap_block = []
    
    def backward_hook(module, grad_in, grad_out):
        grad_block.append(grad_out[0].detach())

    def forward_hook(module, input, output):
        fmap_block.append(output)


    def cam_show_img(img, feature_map, grads, out_dir):
        H, W = img.shape[:2]
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        grads = grads.reshape(grads.shape[0], -1)
        weights = np.mean(grads, axis=1)
        for i, w in enumerate(weights):
            cam += w*feature_map[i,:,:]
        cam = np.maximum(cam, 0)
        cam = cam/cam.max()
        cam = cv2.resize(cam, (W, H))

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        cam_img = 0.3*heatmap+0.7*img

        cv2.imwrite(out_dir, cam_img)

    
    test_loader = tqdm(test_loader, ncols=100)
    model_agent.eval()

    model_agent.point_prompt[2].register_forward_hook(forward_hook)
    model_agent.point_prompt[2].register_backward_hook(backward_hook)

    criterion = DiceLoss()

    for batch, batched_input in enumerate(test_loader):
        if batch>100:
            break
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        
        
        labels = batched_input["label"]
        image_input = batched_input['image']
        # labels_edge = batched_input['edge']


        ## step 2 model agent for prompt
        for n, value in model_agent.named_parameters():
            if "prompt" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False
        with torch.no_grad():
            image_embeddings = model_sam.image_encoder(image_input).detach()
            image_embeddings_ori = image_embeddings.clone()
        out = model_agent(image_input, image_embeddings)
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


        # with torch.no_grad():
        # sparse_embeddings  (b*3)*n*256
        # dense_embeddings   (b*3)*256*16*16
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



        

        
        
        mask_fusion = F.sigmoid(mask_sam)*(1-args.fusion_ratio) + F.sigmoid(masks[-1])*args.fusion_ratio
        

        loss_mask = criterion(mask_fusion, labels)
        loss_mask.backward(retain_graph=True)

        grads_val = grad_block[-1].cpu().data.numpy().squeeze()
        fmap = fmap_block[-1].cpu().data.numpy().squeeze()

        image_input = image_input.permute(0,2,3,1).cpu().numpy()[0]
        
        temp_save_dir = os.path.join(args.work_dir, "results_vis", str(args.num_sample), args.model_name+'_m3')
        os.makedirs(temp_save_dir, exist_ok=True)
        save_path_heatmap = os.path.join(temp_save_dir, f"heatmap_{str(batched_input['name'][0])}.jpg")

        save_path_image = os.path.join(temp_save_dir, f"image_{str(batched_input['name'][0])}.jpg")
        save_path_label = os.path.join(temp_save_dir, f"label_{str(batched_input['name'][0])}.jpg")
        save_path_pred_agent = os.path.join(temp_save_dir, f"pred_agent_{str(batched_input['name'][0])}.jpg")
        save_path_pred_sam = os.path.join(temp_save_dir, f"pred_sam_{str(batched_input['name'][0])}.jpg")
        save_path_pred_fusion = os.path.join(temp_save_dir, f"pred_fusion_{str(batched_input['name'][0])}_{args.model_name}.jpg")

        
        
        cam_show_img(image_input, fmap, grads_val, save_path_heatmap)

        cv2.imwrite(save_path_image, image_input)
        temp_label = labels.permute(0,2,3,1).cpu().numpy()[0]
        temp_label = np.concatenate([temp_label, np.zeros_like(temp_label[:,:,0:1])], axis=-1)
        plt.imsave(save_path_label, temp_label)

        temp_pred_agent = F.sigmoid(masks[-1]).permute(0,2,3,1).cpu().numpy()[0]
        temp_pred_agent[temp_pred_agent>0.5] = 1
        temp_pred_agent[temp_pred_agent<1] = 0
        temp_pred_agent = np.concatenate([temp_pred_agent, np.zeros_like(temp_pred_agent[:,:,0:1])], axis=-1)
        plt.imsave(save_path_pred_agent, temp_pred_agent)

        temp_pred_sam = F.sigmoid(mask_sam.detach()).permute(0,2,3,1).cpu().numpy()[0]
        temp_pred_sam[temp_pred_sam>0.5] = 1
        temp_pred_sam[temp_pred_sam<1] = 0
        temp_pred_sam = np.concatenate([temp_pred_sam, np.zeros_like(temp_pred_sam[:,:,0:1])], axis=-1)
        plt.imsave(save_path_pred_sam, temp_pred_sam)

        temp_pred_fusion = mask_fusion.detach().permute(0,2,3,1).cpu().numpy()[0]
        temp_pred_fusion[temp_pred_fusion>0.5] = 1
        temp_pred_fusion[temp_pred_fusion<1] = 0
        temp_pred_fusion = np.concatenate([temp_pred_fusion, np.zeros_like(temp_pred_fusion[:,:,0:1])], axis=-1)
        plt.imsave(save_path_pred_fusion, temp_pred_fusion)


def vis_models(args):
    if args.model_name == 'Unet':
        model = NestedUNet(2, 3, True).to(args.device)
    elif args.model_name == 'FCN':
        model = fcn_resnet50_my(num_classes=2).to(args.device)
    elif args.model_name == 'DeepLabv3':
        model = deeplabv3_resnet50_my(num_classes=2, model_type='m0').to(args.device)
    elif args.model_name == 'PSPNet':
        model = PSPNet(n_classes=2).to(args.device)
    elif args.model_name == 'Fast-SCNN':
        model = FastSCNN(num_classes=2).to(args.device)
    elif args.model_name == 'SegFormer':
        model = Segformer(num_classes=2).to(args.device)
    elif args.model_name == 'TGANet':
        model = TGAPolypSeg(num_classes=2).to(args.device)
    elif args.model_name == 'VM':

        # model_config = {
        # 'num_classes': 9, 
        # 'input_channels': 3, 
        # # ----- VM-UNet ----- #
        # 'depths': [2,2,2,2],
        # 'depths_decoder': [2,2,2,1],
        # 'drop_path_rate': 0.2,
        # 'load_ckpt_path': './pre_trained_weights/vmamba_small_e238_ema.pth',
        # }

        model = VMUNet(input_channels=3, num_classes=2,
                       depths=[2,2,2,2],
                       depths_decoder=[2,2,2,1],
                       drop_path_rate=0.2,
                       ).to(args.device)

    
    
    test_dataset = TrainingDataset_unet(args.data_path, image_size=args.image_size, mode='test', requires_name = False, fold=args.fold, num_sample=-1, num_fold=10)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)
    print('*******test data:', len(test_dataset)) 



    temp_save_dir = os.path.join(args.work_dir, "models", str(args.num_sample), args.model_name)
    os.makedirs(temp_save_dir, exist_ok=True)
    save_path = os.path.join(temp_save_dir, f"temp_{args.model_name}_fold{str(args.fold)}.pth")
    ck = torch.load(save_path)['model']
    model.load_state_dict(ck)

    
    grad_block = []
    fmap_block = []
    
    def backward_hook(module, grad_in, grad_out):
        grad_block.append(grad_out[0].detach())

    def forward_hook(module, input, output):
        fmap_block.append(output)


    def cam_show_img(img, feature_map, grads, out_dir):
        H, W = img.shape[:2]
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        grads = grads.reshape(grads.shape[0], -1)
        weights = np.mean(grads, axis=1)
        for i, w in enumerate(weights):
            cam += w*feature_map[i,:,:]
        cam = np.maximum(cam, 0)
        cam = cam/cam.max()
        cam = cv2.resize(cam, (W, H))

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        cam_img = 0.3*heatmap+0.7*img

        cv2.imwrite(out_dir, cam_img)

    
    test_loader = tqdm(test_loader, ncols=100)
    model.eval()

    # model_agent.point_prompt[3].register_forward_hook(forward_hook)
    # model_agent.point_prompt[3].register_backward_hook(backward_hook)

    criterion = DiceLoss()

    for batch, batched_input in enumerate(test_loader):
        if batch>100:
            break
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        
        
        labels = batched_input["label"]
        image_input = batched_input['image']
        # labels_edge = batched_input['edge']


        with torch.no_grad():
            out = model(image_input)
        if args.model_name == 'Unet':
            masks = out['masks']
        # else args.model_name == 'FCN' or args.model_name == 'DeepLabv3' or args.model_name == 'PSPNet' or args.model_name == 'Fast-SCNN' or args.model_name == 'TGANet':
        else:
            mask = out['out']
            masks = [mask]

        
        



        

        

        # loss_mask = criterion(F.sigmoid(masks[-1]), labels)
        # loss_mask.backward(retain_graph=False)

        # grads_val = grad_block[-1].cpu().data.numpy().squeeze()
        # fmap = fmap_block[-1].cpu().data.numpy().squeeze()

        image_input = image_input.permute(0,2,3,1).cpu().numpy()[0]
        
        temp_save_dir = os.path.join(args.work_dir, "results_vis", str(args.num_sample), args.model_name)
        os.makedirs(temp_save_dir, exist_ok=True)
        save_path_heatmap = os.path.join(temp_save_dir, f"heatmap_{str(batched_input['name'][0])}.jpg")

        save_path_image = os.path.join(temp_save_dir, f"image_{str(batched_input['name'][0])}.jpg")
        save_path_label = os.path.join(temp_save_dir, f"label_{str(batched_input['name'][0])}.jpg")
        save_path_pred_agent = os.path.join(temp_save_dir, f"pred_agent_{str(batched_input['name'][0])}_{args.model_name}.jpg")

        
        
        # cam_show_img(image_input, fmap, grads_val, save_path_heatmap)

        cv2.imwrite(save_path_image, image_input)
        temp_label = labels.permute(0,2,3,1).cpu().numpy()[0]
        temp_label = np.concatenate([temp_label, np.zeros_like(temp_label[:,:,0:1])], axis=-1)

        plt.imsave(save_path_label, temp_label)

        temp_pred_agent = F.sigmoid(masks[-1].detach()).permute(0,2,3,1).cpu().numpy()[0]
        temp_pred_agent[temp_pred_agent>0.5] = 1
        temp_pred_agent[temp_pred_agent<1] = 0

        temp_pred_agent = np.concatenate([temp_pred_agent, np.zeros_like(temp_pred_agent[:,:,0:1])], axis=-1)
        plt.imsave(save_path_pred_agent, temp_pred_agent)

from models import sam_seg_model_registry
def vis_autosam(args):
    args.model_name = 'autosam'

    model = sam_seg_model_registry['vit_b'](num_classes=2, checkpoint=None).to(args.device)


    test_dataset = TrainingDataset_unet(args.data_path, image_size=args.image_size, mode='test', requires_name = False, fold=args.fold, num_sample=-1, num_fold=10)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=0)
    print('*******test data:', len(test_dataset)) 

 
    temp_save_dir = os.path.join(args.work_dir, "models", str(args.num_sample), args.model_name)
    os.makedirs(temp_save_dir, exist_ok=True)
    save_path = os.path.join(temp_save_dir, f"temp_{args.model_name}_fold{str(args.fold)}.pth")
    ck = torch.load(save_path)['model']
    model.load_state_dict(ck)




    
    
    test_dataset = TrainingDataset_unet(args.data_path, image_size=args.image_size, mode='test', requires_name = False, fold=args.fold, num_sample=-1, num_fold=10)
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=0)
    print('*******test data:', len(test_dataset)) 



    temp_save_dir = os.path.join(args.work_dir, "models", str(args.num_sample), args.model_name)
    os.makedirs(temp_save_dir, exist_ok=True)
    save_path = os.path.join(temp_save_dir, f"temp_{args.model_name}_fold{str(args.fold)}.pth")
    ck = torch.load(save_path)['model']
    model.load_state_dict(ck)

    
    grad_block = []
    fmap_block = []
    
    def backward_hook(module, grad_in, grad_out):
        grad_block.append(grad_out[0].detach())

    def forward_hook(module, input, output):
        fmap_block.append(output)


    def cam_show_img(img, feature_map, grads, out_dir):
        H, W = img.shape[:2]
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        grads = grads.reshape(grads.shape[0], -1)
        weights = np.mean(grads, axis=1)
        for i, w in enumerate(weights):
            cam += w*feature_map[i,:,:]
        cam = np.maximum(cam, 0)
        cam = cam/cam.max()
        cam = cv2.resize(cam, (W, H))

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        cam_img = 0.3*heatmap+0.7*img

        cv2.imwrite(out_dir, cam_img)

    
    test_loader = tqdm(test_loader, ncols=100)
    model.eval()

    # model_agent.point_prompt[3].register_forward_hook(forward_hook)
    # model_agent.point_prompt[3].register_backward_hook(backward_hook)

    criterion = DiceLoss()

    for batch, batched_input in enumerate(test_loader):
        if batch>100:
            break
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        
        
        labels = batched_input["label"]
        image_input = batched_input['image']
        # labels_edge = batched_input['edge']



        with torch.no_grad():
            out = model(image_input)
        b,c,h,w = labels.shape
        out = model(image_input)

        out = out[0].view(b,2,h,w)
        masks = [out]
        
        
        



        

        

        # loss_mask = criterion(F.sigmoid(masks[-1]), labels)
        # loss_mask.backward(retain_graph=False)

        # grads_val = grad_block[-1].cpu().data.numpy().squeeze()
        # fmap = fmap_block[-1].cpu().data.numpy().squeeze()

        image_input = image_input.permute(0,2,3,1).cpu().numpy()[0]
        
        temp_save_dir = os.path.join(args.work_dir, "results_vis", str(args.num_sample), args.model_name)
        os.makedirs(temp_save_dir, exist_ok=True)
        save_path_heatmap = os.path.join(temp_save_dir, f"heatmap_{str(batched_input['name'][0])}.jpg")

        save_path_image = os.path.join(temp_save_dir, f"image_{str(batched_input['name'][0])}.jpg")
        save_path_label = os.path.join(temp_save_dir, f"label_{str(batched_input['name'][0])}.jpg")
        save_path_pred_agent = os.path.join(temp_save_dir, f"pred_agent_{str(batched_input['name'][0])}_{args.model_name}.jpg")

        
        
        # cam_show_img(image_input, fmap, grads_val, save_path_heatmap)

        cv2.imwrite(save_path_image, image_input)
        temp_label = labels.permute(0,2,3,1).cpu().numpy()[0]
        temp_label = np.concatenate([temp_label, np.zeros_like(temp_label[:,:,0:1])], axis=-1)
        plt.imsave(save_path_label, temp_label)

        temp_pred_agent = F.sigmoid(masks[-1].detach()).permute(0,2,3,1).cpu().numpy()[0]
        temp_pred_agent[temp_pred_agent>0.5] = 1
        temp_pred_agent[temp_pred_agent<1] = 0
        temp_pred_agent = np.concatenate([temp_pred_agent, np.zeros_like(temp_pred_agent[:,:,0:1])], axis=-1)
        plt.imsave(save_path_pred_agent, temp_pred_agent)

def vis_nnsam(args):
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



    temp_save_dir = os.path.join(args.work_dir, "models", str(args.num_sample), args.model_name+'_m1')
    os.makedirs(temp_save_dir, exist_ok=True)
    save_path = os.path.join(temp_save_dir, f"temp_{args.model_name+'_m1'}_temp.pth")
    args.sam_checkpoint = save_path
    ck = torch.load(save_path)['model']
    model_agent.load_state_dict(ck)

    model_sam = sam_model_registry[args.model_type](args).to(args.device)

    
    grad_block = []
    fmap_block = []
    
    def backward_hook(module, grad_in, grad_out):
        grad_block.append(grad_out[0].detach())

    def forward_hook(module, input, output):
        fmap_block.append(output)


    def cam_show_img(img, feature_map, grads, out_dir):
        H, W = img.shape[:2]
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        grads = grads.reshape(grads.shape[0], -1)
        weights = np.mean(grads, axis=1)
        for i, w in enumerate(weights):
            cam += w*feature_map[i,:,:]
        cam = np.maximum(cam, 0)
        cam = cam/cam.max()
        cam = cv2.resize(cam, (W, H))

        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        cam_img = 0.3*heatmap+0.7*img

        cv2.imwrite(out_dir, cam_img)

    
    test_loader = tqdm(test_loader, ncols=100)
    model_agent.eval()

    model_agent.point_prompt[2].register_forward_hook(forward_hook)
    model_agent.point_prompt[2].register_backward_hook(backward_hook)

    criterion = DiceLoss()

    for batch, batched_input in enumerate(test_loader):
        if batch>100:
            break
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        
        
        labels = batched_input["label"]
        image_input = batched_input['image']
        # labels_edge = batched_input['edge']


        ## step 2 model agent for prompt
        for n, value in model_agent.named_parameters():
            if "prompt" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False
        with torch.no_grad():
            image_embeddings = model_sam.image_encoder(image_input).detach()
            image_embeddings_ori = image_embeddings.clone()
        out = model_agent(image_input, image_embeddings)
        if args.model_name == 'Unet' or args.model_name == 'Unet_sam' or args.model_name == 'Unet_sam_joint':
            masks = out['masks']
        elif args.model_name == 'FCN' or args.model_name == 'DeepLabv3' or args.model_name == 'PSPNet' or args.model_name == 'Fast-SCNN' or args.model_name == 'TGANet':
            mask = out['out']
            masks = [mask]

        
    
        

        # loss_mask = criterion(F.sigmoid(masks[-1]), labels)
        # loss_mask.backward(retain_graph=True)

        # grads_val = grad_block[-1].cpu().data.numpy().squeeze()
        # fmap = fmap_block[-1].cpu().data.numpy().squeeze()

        image_input = image_input.permute(0,2,3,1).cpu().numpy()[0]
        
        temp_save_dir = os.path.join(args.work_dir, "results_vis", str(args.num_sample), args.model_name+'_m1')
        os.makedirs(temp_save_dir, exist_ok=True)
        # save_path_heatmap = os.path.join(temp_save_dir, f"heatmap_{str(batched_input['name'][0])}.jpg")

        save_path_image = os.path.join(temp_save_dir, f"image_{str(batched_input['name'][0])}.jpg")
        save_path_label = os.path.join(temp_save_dir, f"label_{str(batched_input['name'][0])}.jpg")
        save_path_pred_agent = os.path.join(temp_save_dir, f"pred_nnsam_{str(batched_input['name'][0])}_{args.model_name}.jpg")


        
        
        # cam_show_img(image_input, fmap, grads_val, save_path_heatmap)

        cv2.imwrite(save_path_image, image_input)
        temp_label = labels.permute(0,2,3,1).cpu().numpy()[0]
        temp_label = np.concatenate([temp_label, np.zeros_like(temp_label[:,:,0:1])], axis=-1)
        plt.imsave(save_path_label, temp_label)

        temp_pred_agent = F.sigmoid(masks[-1]).permute(0,2,3,1).cpu().numpy()[0]
        temp_pred_agent[temp_pred_agent>0.5] = 1
        temp_pred_agent[temp_pred_agent<1] = 0
        temp_pred_agent = np.concatenate([temp_pred_agent, np.zeros_like(temp_pred_agent[:,:,0:1])], axis=-1)
        plt.imsave(save_path_pred_agent, temp_pred_agent)



if __name__ == '__main__':
    args = parse_args()
    args.fold = 0
    

    # vis_models(args)


    # list_num_samples = [1,4,8,16]
    # for temp_num in list_num_samples:
    #     args.num_sample = temp_num
    #     vis_autosam(args)


    # ist_model_names = ['FCN', 'DeepLabv3',]
    # list_num_samples = [1,4,8,16]
    # for model_name in ist_model_names:
    #     args.model_name = model_name
    #     for temp_num in list_num_samples:
    #         args.num_sample = temp_num
    #         vis_nnsam(args)
    

    ist_model_names = ['FCN', 'DeepLabv3',]
    list_num_samples = [1,4,8,16]
    for model_name in ist_model_names:
        args.model_name = model_name
        for temp_num in list_num_samples:
            args.num_sample = temp_num
            vis(args)
    
