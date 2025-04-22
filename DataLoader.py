
import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_transforms, get_boxes_from_mask, init_point_sampling, valid_transforms
import json
import random
import matplotlib.pyplot as plt

import torchvision.transforms

class TestingDataset(Dataset):
    
    def __init__(self, data_path, image_size=256, mode='test', requires_name=True, point_num=1, return_ori_mask=True, prompt_path=None, fold=0):
        """
        Initializes a TestingDataset object.
        Args:
            data_path (str): The path to the data.
            image_size (int, optional): The size of the image. Defaults to 256.
            mode (str, optional): The mode of the dataset. Defaults to 'test'.
            requires_name (bool, optional): Indicates whether the dataset requires image names. Defaults to True.
            point_num (int, optional): The number of points to retrieve. Defaults to 1.
            return_ori_mask (bool, optional): Indicates whether to return the original mask. Defaults to True.
            prompt_path (str, optional): The path to the prompt file. Defaults to None.
        """
        self.image_size = image_size
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        self.prompt_list = {} if prompt_path is None else json.load(open(prompt_path, "r"))
        self.requires_name = requires_name
        self.point_num = point_num

        json_file = open(os.path.join(data_path, f'label2image_{mode}_fold{str(fold)}.json'), "r")
        dataset = json.load(json_file)
    
        self.image_paths = list(dataset.values())
        self.label_paths = list(dataset.keys())
      
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]
    
    def __getitem__(self, index):
        """
        Retrieves and preprocesses an item from the dataset.
        Args:
            index (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the preprocessed image and associated information.
        """
        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            # image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        mask_path = self.label_paths[index]
        ori_np_mask = cv2.imread(mask_path, 0)

        
        if ori_np_mask.max() >200:
            ori_np_mask[ori_np_mask<200] = 0
            ori_np_mask[ori_np_mask>0] = 1

        assert np.array_equal(ori_np_mask, ori_np_mask.astype(bool)), f"Mask should only contain binary values 0 and 1. {self.label_paths[index]}"

        h, w = ori_np_mask.shape
        ori_mask = torch.tensor(ori_np_mask).unsqueeze(0)

        transforms = valid_transforms(self.image_size, h, w)
        augments = transforms(image=image, mask=ori_np_mask)
        image, mask = augments['image'], augments['mask'].to(torch.int64)
        cls_list = []
        if self.prompt_path is None:
            # boxes = get_boxes_from_mask(mask)
            point_coords, point_labels = init_point_sampling(mask, self.point_num)
        else:
            prompt_key = mask_path.split('/')[-1]
            # boxes = torch.as_tensor(self.prompt_list[prompt_key]["boxes"], dtype=torch.float)
            point_coords = torch.as_tensor(self.prompt_list[prompt_key]["point_coords"], dtype=torch.float)
            point_labels = torch.as_tensor(self.prompt_list[prompt_key]["point_labels"], dtype=torch.int)

        temp_cls = int(mask_path.split('_')[-2])
        cls_list.append(torch.from_numpy(np.array(temp_cls)))
        cls_s = torch.stack(cls_list, dim=0)

        mask[mask>=0.5] = 1
        mask[mask<1] = 0
        image_input["image"] = image
        image_input["label"] = mask.unsqueeze(0)
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        # image_input["boxes"] = boxes
        image_input["original_size"] = (h, w)
        image_input["label_path"] = '/'.join(mask_path.split('/')[:-1])
        image_input['cls'] = cls_s


        if self.return_ori_mask:
            image_input["ori_label"] = ori_mask
     
        image_name = self.label_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.label_paths)



def rotate_img(img, angle, scale=1.0, dx=0, dy=0):
    '''
    img   --image
    angle --rotation angle
    return--rotated img
    '''
    h, w = img.shape[:2]
    rotate_center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(rotate_center, angle, scale)
    # #计算图像新边界
    # new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    # new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
    # #调整旋转矩阵以考虑平移
    M[0, 2] += dx
    M[1, 2] += dy

    rotated_img = cv2.warpAffine(img, M, (w, h), borderValue=0)
    # rotated_img = cv2.resize(rotated_img, dsize=(w, h))
    return rotated_img


def adjust_brightness_contrast(image, alpha=1.0, beta=0.0):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def random_brightness_contrast(image, brightness_range=(-30, 30), contrast_range=(0.8, 1.2)):
    """
    随机调整图像的亮度和对比度。
    
    Parameters:
    - image: 输入图像
    - brightness_range: 亮度调整范围，例如 (-30, 30)
    - contrast_range: 对比度缩放范围，例如 (0.8, 1.2)
    
    Returns:
    - 调整后的图像
    """
    # 随机生成亮度和对比度的调整参数
    random_brightness = np.random.randint(brightness_range[0], brightness_range[1] + 1)
    random_contrast = np.random.uniform(contrast_range[0], contrast_range[1])
    
    # 调整亮度和对比度
    adjusted_image = adjust_brightness_contrast(image, alpha=random_contrast, beta=random_brightness)
    
    return adjusted_image


def flip_img(img, flag_flip):
    if flag_flip < 2:
        imgdst = cv2.flip(img, flag_flip)
    else:
        imgdst = img
    return imgdst

class TrainingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5, fold=0):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [123.675, 116.28, 103.53]
        self.pixel_std = [58.395, 57.12, 57.375]

        dataset = json.load(open(os.path.join(data_dir, f'image2label_train_fold{str(fold)}.json'), "r"))
        self.image_paths = list(dataset.keys())
        num_split = int(len(self.image_paths)*0.1)
        if mode=='train':
            self.image_paths = self.image_paths[num_split:]
        elif mode=='valid':
            self.image_paths = self.image_paths[:num_split]
        else:
            pass
        self.dataset = dataset
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            # 
        except:
            print(self.image_paths[index])

        # flag_flip = random.choice([-1,0,1,2,2,2])
        # image = flip_img(image, flag_flip)
        
        # if random.random() < 0.5:
            # image = random_brightness_contrast(image, (-20 ,20), (0.75,1.25))
            # image = cv2.GaussianBlur(image, ksize=(3,3), sigmaX=3)


        random_angle = random.randint(-10, 10)

        # random_scale = np.random.randint(50,110)/100
        random_scale = 1.0
        # r_dx = random.randint(-30, 30)
        # r_dy = random.randint(-30, 30)
        r_dx = 0
        r_dy = 0
        # image = rotate_img(image, random_angle, random_scale, r_dx, r_dy)

        # fig = plt.figure()
        # plt.imshow(image)
        # plt.show()
        
        
        # image = (image - self.pixel_mean) / self.pixel_std
        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        masks_pt_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        cls_list = []
        # mask_path = random.choices(self.label_paths[index], k=self.mask_num)
        # mask_path = self.label_paths[index]*(self.mask_num//3)
        mask_path = self.dataset[self.image_paths[index]]
        random.shuffle(mask_path)
        for m in mask_path:
            temp_cls = int(m.split('_')[-2])
            cls_list.append(torch.from_numpy(np.array(temp_cls)))
            pre_mask = cv2.imread(m, 0)

            # pre_mask = flip_img(pre_mask, flag_flip)
            # pre_mask = rotate_img(pre_mask, random_angle, random_scale, r_dx, r_dy)

            # fig = plt.figure()
            # plt.imshow(pre_mask)
            # plt.show()


            if pre_mask.max() >200:
                pre_mask[pre_mask<200] = 0
                pre_mask[pre_mask>0] = 1
            
            mask_for_pt = random_mask(pre_mask)
            # mask_for_pt = pre_mask
            if mask_for_pt.max() > 200:
                mask_for_pt[mask_for_pt<200] = 0
                mask_for_pt[mask_for_pt>0] = 1

            
            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].long()
            augments = transforms(image=image, mask=mask_for_pt)
            mask_pt_tensor = augments['mask'].long()

            # boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            masks_pt_list.append(mask_pt_tensor)
            # boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        mask[mask>=0.5] = 1
        mask[mask<1] = 0
        mask_pt = torch.stack(masks_pt_list, dim=0)
        mask_pt[mask_pt>=0.5] = 1
        mask_pt[mask_pt<1] = 0
        # boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)
        cls_s = torch.stack(cls_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        image_input['label_pt'] = mask_pt.unsqueeze(1)
        # image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input['cls'] = cls_s
        image_input['mask_path'] = mask_path

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)


def stack_dict_batched(batched_input):
    out_dict = {}
    for k,v in batched_input.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict





from convert_dataset import extract_mask_info, load_img_resize


def extract_edge_info(mask):
    temp_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
    mask_edge = []
    dim_num = len(mask.shape)
    if dim_num == 3:
        for i in range(mask.shape[-1]):
            temp_m = mask[:,:,i]
            temp_erode = cv2.morphologyEx(temp_m, cv2.MORPH_ERODE, temp_kernel)
            temp_edge = temp_m-temp_erode 
            mask_edge.append(temp_edge)
    else:
        temp_m = mask
        temp_erode = cv2.morphologyEx(temp_m, cv2.MORPH_ERODE, temp_kernel)
        temp_edge = temp_m-temp_erode 
        mask_edge.append(temp_edge)
    mask_edge = np.stack(mask_edge, axis=-1)

    return mask_edge


def random_mask(mask):
    ksize1 = random.choice([5,7,11,13])
    ksize2 = random.choice([5,7,11,13])
    temp_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize1, ksize2))
    flag = random.choice([-1,1])
    if flag < 0:
        temp_mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, temp_kernel)
    elif flag == 0:
        temp_mask = mask
    else:
        temp_mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, temp_kernel)
    return temp_mask



def average_group(lst, num_groups):
    group_size = len(lst)//num_groups
    result = []
    for i in range(0, len(lst), group_size):
        result.append(lst[i:i + group_size])
    return result



class TrainingDataset_unet(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, fold=0, num_sample=-1, num_fold=10):
        self.image_size = image_size
        self.requires_name = requires_name
        # self.pixel_mean = [123.675, 116.28, 103.53]
        # self.pixel_std = [58.395, 57.12, 57.375]

        dataset = json.load(open(os.path.join(data_dir, f'data.json'), "r"))
        self.image_paths_train = list(dataset.keys())[1100:]
        self.image_paths_test = list(dataset.keys())[:1100]


        if mode=='train':
            self.image_paths = self.image_paths_train[:50][:num_sample]  #2,4,8,16,32
        elif mode=='valid':
            self.image_paths = self.image_paths_train[50:]
        elif mode=='test':
            self.image_paths = self.image_paths_test
        else:
            pass
        self.mode = mode

        if len(self.image_paths) < 500 and mode=='train':
            l = len(self.image_paths)
            self.image_paths = self.image_paths * int(128/l)
    
    def __getitem__(self, index):
        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            # image_name = self.image_paths[index]
        except:
            print(self.image_paths[index])
        
        image_name = self.image_paths[index].split('/')[-1].split('.')[0]
        mask_path = './data/masks/' + image_name + '.bmp'
        mask_data = extract_mask_info(mask_path)
        mask = np.stack([mask_data['1'], mask_data['2']], axis=-1)
        # mask_edge = extract_edge_info(mask)

        # cv2.imwrite('./temp/temp_ori.png', image)
        # cv2.imwrite('./temp/temp_m_ori.png', mask)
        # cv2.imwrite('./temp/temp_e_ori.png', mask_edge)


        # flag_flip = random.choice([-1,0,1,2,2,2])
        # image = flip_img(image, flag_flip)

        # if random.random() < 0.5:
        #     

        if self.mode == 'train':
            random_angle = random.randint(-10, 10)
            random_scale = 1
            # random_scale = np.random.randint(25,125)/100
            r_dx = 0
            r_dy = 0
            image = rotate_img(image, random_angle, random_scale, r_dx, r_dy)
            mask = rotate_img(mask, random_angle, random_scale, r_dx, r_dy)



            # random_angle = random.randint(-10, 10)
            # random_scale = np.random.randint(50,125)/100
            # r_dx = random.randint(-25, 25)
            # r_dy = random.randint(-25, 25)
            # image = rotate_img(image, random_angle, random_scale, r_dx, r_dy)
            # mask = rotate_img(mask, random_angle, random_scale, r_dx, r_dy)



            # image = random_brightness_contrast(image, (-50,50), (0.5,1.5))

            # plt.imshow(mask)
            # plt.show()

        # mask_edge = flip_img(mask_edge, flag_flip)
        # mask_edge = rotate_img(mask_edge, random_angle, random_scale, r_dx, r_dy)

        # cv2.imwrite('./temp/temp.png', image)
        # cv2.imwrite('./temp/temp_m.png', mask)
        # cv2.imwrite('./temp/temp_e.png', mask_edge)

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
        mask[mask<150] = 0
        mask[mask>100] = 1
        # mask_edge[mask_edge<150] = 0
        # mask_edge[mask_edge>100] = 1

        augments = transforms(image=image, mask=mask)

        image_tensor, mask_tensor = augments['image'], augments['mask']
        # edge_tensor = transforms(image=image, mask=mask_edge)['mask']

        mask_tensor[mask_tensor>=0.5] = 1
        mask_tensor[mask_tensor<1] = 0
        # edge_tensor[edge_tensor>=0.5] = 1
        # edge_tensor[edge_tensor<1] = 0


        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = torch.permute(mask_tensor, dims=(2,0,1)).unsqueeze(0)
        # image_input['edge'] = torch.permute(edge_tensor, dims=(2,0,1)).unsqueeze(0)
        image_input['name'] = image_name

        return image_input
    def __len__(self):
        return len(self.image_paths)


class TestingDataset_unet(Dataset):
    def __init__(self, data_path, image_size=256, mode='test', requires_name=True, fold=0):
        self.image_size = image_size
        self.requires_name = requires_name

        json_file = open(os.path.join(data_path, f'label2image_{mode}_fold{str(fold)}.json'), "r")
        dataset = json.load(json_file)
    
        self.image_paths = list(dataset.values())
        self.label_paths = list(dataset.keys())
        self.dataset = dataset
    
    def __getitem__(self, index):
        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
        except:
            print(self.image_paths[index])

        
        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)


        image_name = self.image_paths[index].split('/')[-1].split('.')[0]
        mask_path = './data/masks/' + image_name + '.bmp'

        mask_data = extract_mask_info(mask_path)
        mask = np.stack([mask_data['1'], mask_data['2'], mask_data['3']], axis=-1)
        mask[mask<150] = 0
        mask[mask>100] = 1
        
        augments = transforms(image=image, mask=mask)

        image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)


        mask_tensor[mask_tensor>=0.5] = 1
        mask_tensor[mask_tensor<1] = 0
        image_input["image"] = image_tensor
        image_input["label"] = torch.permute(mask_tensor, dims=(2,0,1))


        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name

        return image_input
    def __len__(self):
        return len(self.image_paths)








if __name__ == "__main__":
    train_dataset = TrainingDataset("data", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    print("Dataset:", len(train_dataset))
    train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=4)
    for i, batched_image in enumerate(tqdm(train_batch_sampler)):
        batched_image = stack_dict_batched(batched_image)
        print(batched_image["image"].shape, batched_image["label"].shape)

