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
import cv2
import warnings
warnings.filterwarnings('ignore')

from model_s import NestedUNet, PSPNet, FastSCNN, TGAPolypSeg
# from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50
from model_deeplab import deeplabv3_resnet50_my
from model_fcn import fcn_resnet50_my
from segformer_pytorch.segformer_pytorch import Segformer

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
from models_vm.vmunet import VMUNet


resume_dir = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir_fold10", help="work dir")
    # parser.add_argument("--work_dir", type=str, default="workdir_aug", help="work dir")

    parser.add_argument("--data_path", type=str, default="data/fold_10/", help="train data path")
    # parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    # parser.add_argument("--data_path", type=str, default="data/fold_10/", help="train data path")


    parser.add_argument("--run_name", type=str, default="others", help="run model name")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    
    parser.add_argument("--metrics", nargs='+', default=['dice', 'dice_1','dice_2',], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--model_name", type=str, default="DeepLabv3", help="model names [FCN, DeepLab, PSPNet, Fast-SCNN, SegFormer, Mask2Former, TGANet, Unet]")
    parser.add_argument("--cls_emb", type=bool, default=False, help="use class embedding")
    parser.add_argument("--fold", type=int, default=0, help="fold number")
    parser.add_argument("--num_sample", type=int, default=1, help="num of samples") # 2,4,8,16,32
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


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion):
    train_loader = tqdm(train_loader, ncols=100)
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)
    for batch, batched_input in enumerate(train_loader):
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        
        
        labels = batched_input["label"]
        image_input = batched_input['image']
        # labels_edge = batched_input['edge']


        out = model(image_input)

        if args.model_name == 'Unet':
            masks = out['masks']
        elif args.model_name == 'FCN' or 'DeepLabV3' or 'PSPNet' or 'Fast-SCNN' or 'TGANet':
            mask = out['out']
            masks = [mask]


        
        loss_mask = 0
        for m in masks:
            loss_mask += criterion(F.sigmoid(m), labels)

        loss = loss_mask
        loss.backward(retain_graph=False)

        if batch % 8//args.batch_size==0:
            optimizer.step()
            optimizer.zero_grad()


        train_losses.append(loss.item())

        gpu_info = {}
        gpu_info['gpu_name'] = args.device 
        train_loader.set_postfix(train_loss=loss.item())

        train_batch_metrics = SegMetrics(F.sigmoid(masks[-1]), labels, args.metrics)
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]

    return train_losses, train_iter_metrics


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


        with torch.no_grad():
            out = model(image_input)
        if args.model_name == 'Unet':
            masks = out['masks']
        elif args.model_name == 'FCN' or 'DeepLabV3' or 'PSPNet' or 'Fast-SCNN' or 'TGANet':
            mask = out['out']
            masks = [mask]
        


        gpu_info = {}
        gpu_info['gpu_name'] = args.device 


        valid_batch_metrics = SegMetrics(F.sigmoid(masks[-1]), labels, args.metrics)
        valid_iter_metrics = [valid_iter_metrics[i] + valid_batch_metrics[i] for i in range(len(args.metrics))]
        for i in range(len(args.metrics)):
            valid_metrics_all[args.metrics[i]].append(float(valid_batch_metrics[i]))

    model.train()
    return valid_iter_metrics, valid_metrics_all


def train(args):
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


    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4, amsgrad=True)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    criterion = DiceLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs//2,], gamma = 0.1)



    train_dataset = TrainingDataset_unet(args.data_path, image_size=args.image_size, mode='train', requires_name = False, fold=args.fold, num_sample=args.num_sample, num_fold=10)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=0)
    print('*******Train data:', len(train_dataset))   

    valid_dataset = TrainingDataset_unet(args.data_path, image_size=args.image_size, mode='valid', requires_name = False, fold=args.fold, num_sample=-1, num_fold=10)
    valid_loader = DataLoader(valid_dataset, batch_size = 1, shuffle=True, num_workers=0)
    print('*******Valid data:', len(valid_dataset))   

    test_dataset = TrainingDataset_unet(args.data_path, image_size=args.image_size, mode='test', requires_name = False, fold=args.fold, num_sample=-1, num_fold=10)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=True, num_workers=0)
    print('*******test data:', len(test_dataset)) 

    # loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))
    l = len(valid_loader)
    best_metric = 0

    for epoch in range(0, args.epochs):
        model.train()
        start = time.time()

        _, train_iter_metrics = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion)
        train_iter_metrics = [metric / len(train_loader) for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}
        scheduler.step()



        valid_iter_metrics,_ = eval_one_epoch(args, model, valid_loader)
        valid_iter_metrics = [metric / len(valid_loader) for metric in valid_iter_metrics]
        valid_metrics = {args.metrics[i]: '{:.4f}'.format(valid_iter_metrics[i]) for i in range(len(valid_iter_metrics))}
        temp_metic = (float(valid_metrics['dice_1'])+float(valid_metrics['dice_2']))/2
        valid_metrics['dice'] = str('{:.4f}'.format(temp_metic))

        lr = scheduler.get_last_lr()[0]
        print(f"epoch: {epoch + 1}, lr: {lr}, train metrics: {train_metrics}")
        print(f"epoch: {epoch + 1}, lr: {lr}, valid metrics: {valid_metrics}")


        if temp_metic >= best_metric and float(train_metrics['dice'])>0.95 and epoch>10:
            temp_save_dir = os.path.join(args.work_dir, "models", str(args.num_sample), args.model_name)
            os.makedirs(temp_save_dir, exist_ok=True)
            save_path = os.path.join(temp_save_dir, f"temp_{args.model_name}_fold{str(args.fold)}.pth")
            state = {'model': model.float().state_dict(), 'optimizer': optimizer}
            torch.save(state, save_path)
            best_metric = temp_metic
        elif temp_metic >= best_metric and epoch == args.epochs-1:
            temp_save_dir = os.path.join(args.work_dir, "models", str(args.num_sample), args.model_name)
            os.makedirs(temp_save_dir, exist_ok=True)
            save_path = os.path.join(temp_save_dir, f"temp_{args.model_name}_fold{str(args.fold)}.pth")
            state = {'model': model.float().state_dict(), 'optimizer': optimizer}
            torch.save(state, save_path)
            best_metric = temp_metic

        
        end = time.time()
        print("Run epoch time: %.2fs" % (end - start))

    
    temp_save_dir = os.path.join(args.work_dir, "models", str(args.num_sample), args.model_name)
    save_path = os.path.join(temp_save_dir, f"temp_{args.model_name}_fold{str(args.fold)}.pth")
    ck = torch.load(save_path)['model']
    model.load_state_dict(ck)
    test_iter_metrics,_ = eval_one_epoch(args, model, test_loader)
    test_iter_metrics = [metric / len(test_loader) for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}
    temp_metic = (float(test_metrics['dice_1'])+float(test_metrics['dice_2']))/2
    test_metrics['dice'] = str('{:.4f}'.format(temp_metic))
    print(test_metrics)
    return test_metrics


def eval(args):
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



    ##analysis module parameters FLops
    # temp_input = (torch.randn(2,3,256,256).to(args.device))
    # flops = FlopCountAnalysis(model, temp_input)
    # print('FLOPs:{:.2f}*e10'.format(0.5*flops.total()/1e10))
    # print(parameter_count_table(model))


    # return 0
    
    test_dataset = TrainingDataset_unet(args.data_path, image_size=args.image_size, mode='test', requires_name = False, fold=args.fold, num_sample=-1, num_fold=10)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=True, num_workers=0)
    print('*******test data:', len(test_dataset)) 

    best_metric = 0


    temp_save_dir = os.path.join(args.work_dir, "models", str(args.num_sample), args.model_name)
    os.makedirs(temp_save_dir, exist_ok=True)
    save_path = os.path.join(temp_save_dir, f"temp_{args.model_name}_fold{str(args.fold)}.pth")
    ck = torch.load(save_path)['model']
    model.load_state_dict(ck)
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





def vis(args):
    if args.model_name == 'Unet':
        model = NestedUNet(3, 3, True).to(args.device)
    elif args.model_name == 'FCN':
        model = fcn_resnet50(num_classes=3).to(args.device)
    elif args.model_name == 'DeepLab':
        model = deeplabv3_resnet50(num_classes=3).to(args.device)
    elif args.model_name == 'PSPNet':
        model = PSPNet(n_classes=3).to(args.device)
    elif args.model_name == 'Fast-SCNN':
        model = FastSCNN(num_classes=3).to(args.device)
    elif args.model_name == 'SegFormer':
        model = Segformer().to(args.device)
    # elif args.model_name == 'Mask2Former':
    #     pass
    elif args.model_name == 'TGANet':
        model = TGAPolypSeg(num_classes=3).to(args.device)

    
    
    test_dataset = TrainingDataset_unet(args.data_path, image_size=args.image_size, mode='test', requires_name = False, fold=args.fold, num_sample=-1, num_fold=10)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, num_workers=0)
    print('*******test data:', len(test_dataset)) 

    temp_save_dir = os.path.join(args.work_dir, "models", str(args.num_sample), args.model_name)
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

    model.conv0_4.register_forward_hook(forward_hook)
    model.conv0_4.register_backward_hook(backward_hook)

    criterion = DiceLoss()

    for batch, batched_input in enumerate(test_loader):
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)
        
        
        labels = batched_input["label"]
        image_input = batched_input['image']
        # labels_edge = batched_input['edge']


        out = model(image_input)

        

        if args.model_name == 'Unet':
            masks = out['masks']
        elif args.model_name == 'FCN' or 'DeepLabV3' or 'PSPNet' or 'Fast-SCNN' or 'TGANet':
            mask = out['out']
            masks = [mask]
        

        loss_mask = criterion(F.sigmoid(masks[-1]), labels)
        loss_mask.backward(retain_graph=False)

        grads_val = grad_block[0].cpu().data.numpy().squeeze()
        fmap = fmap_block[0].cpu().data.numpy().squeeze()

        image_input = image_input.permute(0,2,3,1).cpu().numpy()[0]
        cam_show_img(image_input, fmap, grads_val, 'test.png')






if __name__ == '__main__':
    args = parse_args()
    if args.train=='true':
        train(args)
    elif args.train=='train':
        list_model_names = ['FCN', 'DeepLabv3', 'PSPNet', 'Fast-SCNN', 'TGANet', 'SegFormer', 'Unet','VM']
        for model_name in list_model_names:
            if model_name == 'VM':
                args.batch_size = 1
            else:
                args.batch_size = 8
            if model_name == 'SegFormer':
                args.lr = 1e-5
            else:
                args.lr = 1e-4
            args.model_name = model_name
            for info in args.__dict__:
                print(info+':'+str(args.__dict__[info]))
            train(args)
    elif args.train =='false':
        args.metrics = ['dice', 'dice_1','dice_2', 'hd_1', 'hd_2', ]
        eval(args)
    elif args.train =='test':
        list_model_names = ['VM']
        # list_model_names = ['FCN', 'DeepLabv3', 'PSPNet', 'Fast-SCNN', 'TGANet', 'SegFormer', 'Unet',]
        args.metrics = ['dice_1','dice_2', 'hd_1', 'hd_2', 
                        'sen_1', 'sen_2', 'spec_1', 'spec_2', 
                        'auc_1', 'auc_2',  'aupr_1', 'aupr_2', 
                        ]
        list_num_samples = [1,2,4,6,8,12,16,20]
        for model_name in list_model_names:
            args.model_name = model_name
            for temp_num in list_num_samples:
                args.num_sample = temp_num
                for info in args.__dict__:
                    print(info+':'+str(args.__dict__[info]))
                eval(args)

    for info in args.__dict__:
        print(info+':'+str(args.__dict__[info]))


    # vis(args)


