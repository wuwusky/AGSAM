import numpy as np
import cv2
import os
import json
from tqdm import tqdm



def load_img_resize(temp_img ,size=256, inter='bi'):
    h, w = temp_img.shape[:2]
    if h > w:
        t = int((h-w)/2)
        b = t
        l = 0
        r = l
    elif w > h:
        t = 0
        b = t
        l = int((w-h)/2)
        r = l
    else:
        t = 0
        b = t
        l = 0
        r = l
    
    temp_img_pad = cv2.copyMakeBorder(src=temp_img, \
                                      top=t, bottom=b, left=l, right=r, \
                                        borderType=cv2.BORDER_CONSTANT, value=0)
    
    # print(temp_img_pad.shape)
    if inter == 'bi':
        temp_img_dst = cv2.resize(temp_img_pad, (size, size))
    else:
        temp_img_dst = cv2.resize(temp_img_pad, (size, size), cv2.INTER_NEAREST)
    # print(temp_img_dst.shape)
    return temp_img_dst

def extract_mask_info(mask_dir):
    temp_mask = cv2.imread(mask_dir, 0)
    # print(temp_mask.max(), temp_mask.min())
    # m1 
    m1 = np.zeros_like(temp_mask)
    m1[temp_mask==0] = 255
    m2 = np.zeros_like(temp_mask)
    m2[temp_mask<150] = 255


    m1 = load_img_resize(m1, 256, 'near')
    m2 = load_img_resize(m2, 256, 'near')

    
    data = {}
    # cv2.imshow('m1', m1)
    # cv2.imshow('m2', m2)
    # cv2.imshow('m3', m3)
    # cv2.waitKey()
    data['1'] = m1
    data['2'] = m2
    return data

def save_info(data, path):
    with open(path, 'w') as f:
        f.write(json.dumps(data))


def average_group(lst, num_groups):
    group_size = len(lst)//num_groups
    result = []
    for i in range(0, len(lst), group_size):
        result.append(lst[i:i + group_size])
    return result


def convert_data_train(fold=1, num_fold=10):
    data_dir = './data/images/'
    mask_dir = './data/masks/'

    data_save_dir = './data/images_re/'
    mask_save_dir = './data/masks_re/'
    os.makedirs(data_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    list_imgnames = os.listdir(data_dir)[:-100]
    list_imgnames_sub = average_group(list_imgnames, num_fold)


    list_imgnames_train = []
    list_imgnames_valid = []
    for i in range(num_fold):
        if i != fold:
            list_imgnames_train += list_imgnames_sub[i]
        else:
            list_imgnames_valid += list_imgnames_sub[i]

    list_imgnames = list_imgnames_train
    train_data_info = {}
    for imgname in tqdm(list_imgnames, ncols=100):
        temp_img_dir = data_dir + imgname
        temp_mask_dir = mask_dir + imgname.split('.')[0] + '.bmp'
        mask_data = extract_mask_info(temp_mask_dir)

        cv2.imwrite(mask_save_dir + imgname.split('.')[0] + '_1_000.png', mask_data['1'])
        cv2.imwrite(mask_save_dir + imgname.split('.')[0] + '_2_000.png', mask_data['2'])



        temp_mask_list = [mask_save_dir + imgname.split('.')[0] + '_1_000.png', \
                          mask_save_dir + imgname.split('.')[0] + '_2_000.png', \
                           ]

        temp_img = cv2.imread(temp_img_dir)
        temp_img_re = load_img_resize(temp_img)
        temp_img_save_dir = data_save_dir + imgname
        cv2.imwrite(temp_img_save_dir, temp_img_re)
        train_data_info[temp_img_save_dir] = temp_mask_list
    temp_save_dir = './data/fold_' + str(num_fold) 
    os.makedirs(temp_save_dir, exist_ok=True)
    save_info(train_data_info, temp_save_dir + '/image2label_train_fold' + str(fold) + '.json')






    list_imgnames = list_imgnames_valid
    valid_data_info = {}
    for imgname in tqdm(list_imgnames, ncols=100):
        temp_img_dir = data_dir + imgname
        temp_mask_dir = mask_dir + imgname.split('.')[0] + '.bmp'
        mask_data = extract_mask_info(temp_mask_dir)

        cv2.imwrite(mask_save_dir + imgname.split('.')[0] + '_1_000.png', mask_data['1'])
        cv2.imwrite(mask_save_dir + imgname.split('.')[0] + '_2_000.png', mask_data['2'])
        cv2.imwrite(mask_save_dir + imgname.split('.')[0] + '_3_000.png', mask_data['3'])


        temp_mask_list = [mask_save_dir + imgname.split('.')[0] + '_1_000.png', \
                          mask_save_dir + imgname.split('.')[0] + '_2_000.png', \
                            mask_save_dir + imgname.split('.')[0] + '_3_000.png']
        
        temp_img = cv2.imread(temp_img_dir)
        temp_img_re = load_img_resize(temp_img)
        temp_img_save_dir = data_save_dir + imgname
        cv2.imwrite(temp_img_save_dir, temp_img_re)
        for t_dir in temp_mask_list:
            valid_data_info[t_dir] = temp_img_save_dir

        
    temp_save_dir = './data/fold_' + str(num_fold)
    os.makedirs(temp_save_dir, exist_ok=True)
    save_info(valid_data_info, temp_save_dir + '/label2image_test_fold' + str(fold) + '.json')

def generate_data_test(num_fold):
    data_dir = './data/images/'
    mask_dir = './data/masks/'

    data_save_dir = './data/images_re/'
    mask_save_dir = './data/masks_re/'
    os.makedirs(data_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    list_imgnames = os.listdir(data_dir)[-160:]

    train_data_info = {}
    for imgname in tqdm(list_imgnames, ncols=100):
        temp_img_dir = data_dir + imgname
        temp_mask_dir = mask_dir + imgname.split('.')[0] + '.bmp'
        mask_data = extract_mask_info(temp_mask_dir)

        cv2.imwrite(mask_save_dir + imgname.split('.')[0] + '_1_000.png', mask_data['1'])
        cv2.imwrite(mask_save_dir + imgname.split('.')[0] + '_2_000.png', mask_data['2'])



        temp_mask_list = [mask_save_dir + imgname.split('.')[0] + '_1_000.png', \
                          mask_save_dir + imgname.split('.')[0] + '_2_000.png', \
                        ]

        temp_img = cv2.imread(temp_img_dir)
        temp_img_re = load_img_resize(temp_img)
        temp_img_save_dir = data_save_dir + imgname
        cv2.imwrite(temp_img_save_dir, temp_img_re)
        train_data_info[temp_img_save_dir] = temp_mask_list
    temp_save_dir = './data/fold_' + str(num_fold) 
    os.makedirs(temp_save_dir, exist_ok=True)
    save_info(train_data_info, temp_save_dir + '/test.json')

def convert_data_valid():
    data_dir = './data/valid_img/'
    mask_dir = './data/valid_mask/'

    data_save_dir = './data/valid_img_re/'
    mask_save_dir = './data/valid_mask_re/'


    list_imgnames = os.listdir(data_dir)

    valid_data_info = {}
    for imgname in tqdm(list_imgnames, ncols=100):
        temp_img_dir = data_dir + imgname
        temp_mask_dir = mask_dir + imgname.split('.')[0] + '.bmp'
        mask_data = extract_mask_info(temp_mask_dir)

        cv2.imwrite(mask_save_dir + imgname.split('.')[0] + '_1_000.png', mask_data['1'])
        cv2.imwrite(mask_save_dir + imgname.split('.')[0] + '_2_000.png', mask_data['2'])
        cv2.imwrite(mask_save_dir + imgname.split('.')[0] + '_3_000.png', mask_data['3'])


        temp_mask_list = [mask_save_dir + imgname.split('.')[0] + '_1_000.png', \
                          mask_save_dir + imgname.split('.')[0] + '_2_000.png', \
                            mask_save_dir + imgname.split('.')[0] + '_3_000.png']
        
        temp_img = cv2.imread(temp_img_dir)
        temp_img_re = load_img_resize(temp_img)
        temp_img_save_dir = data_save_dir + imgname
        cv2.imwrite(temp_img_save_dir, temp_img_re)
        for t_dir in temp_mask_list:
            valid_data_info[t_dir] = temp_img_save_dir

        

    save_info(valid_data_info, './data/label2image_test_new.json')


def convert_train2valid():
    data_dir = './data/image2label_train_new.json'
    data_save_dir = './data/label2image_train_new.json'
    data_info = json.load(open(data_dir, mode='r'))
    
    data_info_new = {}
    for k, v in data_info.items():
        # print(k)
        # print(v)
        for tv in v:
            data_info_new[tv] = k

    save_info(data_info_new, data_save_dir)


def generate_data_all(num_fold):
    data_dir = './data/images/'
    mask_dir = './data/masks/'

    data_save_dir = './data/images_re/'
    mask_save_dir = './data/masks_re/'
    os.makedirs(data_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    list_imgnames = os.listdir(data_dir)[:]

    train_data_info = {}
    for imgname in tqdm(list_imgnames, ncols=100):
        temp_img_dir = data_dir + imgname
        temp_mask_dir = mask_dir + imgname.split('.')[0] + '.bmp'
        mask_data = extract_mask_info(temp_mask_dir)

        cv2.imwrite(mask_save_dir + imgname.split('.')[0] + '_1_000.png', mask_data['1'])
        cv2.imwrite(mask_save_dir + imgname.split('.')[0] + '_2_000.png', mask_data['2'])



        temp_mask_list = [mask_save_dir + imgname.split('.')[0] + '_1_000.png', \
                          mask_save_dir + imgname.split('.')[0] + '_2_000.png', \
]

        temp_img = cv2.imread(temp_img_dir)
        temp_img_re = load_img_resize(temp_img)
        temp_img_save_dir = data_save_dir + imgname
        cv2.imwrite(temp_img_save_dir, temp_img_re)
        train_data_info[temp_img_save_dir] = temp_mask_list
    temp_save_dir = './data/fold_' + str(num_fold) 
    os.makedirs(temp_save_dir, exist_ok=True)
    save_info(train_data_info, temp_save_dir + '/data.json')


if __name__ == '__main__':
    num_fold = 10
    generate_data_all(num_fold)
    generate_data_test(num_fold)
    for i in range(num_fold):
        convert_data_train(i, num_fold)



