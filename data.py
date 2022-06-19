import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance
#from skimage import exposure


def cv_random_flip(img,t,gt,body,detail):
    flip_flag = random.randint(0, 1)
    # flip_flag1= random.randint(0, 1)
    # flip_flag2= random.randint(0, 1)
    
    #left right flip
    if flip_flag == 1:
    # if flip_flag1 == 1:
        img    = img.transpose(Image.FLIP_LEFT_RIGHT)
        t      = t.transpose(Image.FLIP_LEFT_RIGHT)
        gt     = gt.transpose(Image.FLIP_LEFT_RIGHT)
        body   = body.transpose(Image.FLIP_LEFT_RIGHT)
        detail = detail.transpose(Image.FLIP_LEFT_RIGHT)
    
    # #top bottom flip
    # if flip_flag2==1:
    #     img    = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     t      = t.transpose(Image.FLIP_TOP_BOTTOM)
    #     gt     = gt.transpose(Image.FLIP_TOP_BOTTOM)
    #     body   = body.transpose(Image.FLIP_TOP_BOTTOM)
    #     detail = detail.transpose(Image.FLIP_TOP_BOTTOM)
    return img,t,gt,body,detail
def randomCrop(img,t,gt,body,detail):
    border=30
    image_width = img.size[0]
    image_height = img.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return img.crop(random_region), t.crop(random_region),gt.crop(random_region),body.crop(random_region),detail.crop(random_region)
def randomRotation(img,t,gt,body,detail):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        img    = img.rotate(random_angle, mode)
        t      = t.rotate(random_angle, mode)
        gt     = gt.rotate(random_angle, mode)
        body   = body.rotate(random_angle, mode)
        detail = detail.rotate(random_angle, mode)
    return img,t,gt,body,detail
def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
'''def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))'''
def randomPeper(img):
    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):
        randX=random.randint(0,img.shape[0]-1)  
        randY=random.randint(0,img.shape[1]-1)  
        if random.randint(0,1)==0:  
            img[randX,randY]=0  
        else:  
            img[randX,randY]=255 
    return Image.fromarray(img)  

'''def randomGamma(img):
    gamma_flag = random.randint(0, 2)
    if gamma_flag == 1:
        gamma_value = random.uniform(2,5)
        img=np.array(img)
        img = exposure.adjust_gamma(img, gamma_value)
    return Image.fromarray(np.uint8(img))'''

# dataset for training
class SalObjDataset(data.Dataset):
    def __init__(self, train_root, trainsize):
        self.trainsize = trainsize

        self.image_root  = train_root + '/RGB/'
        self.gt_root     = train_root + '/GT/'
        self.t_root      = train_root + '/T/'
        self.body_root   = train_root + '/body/'
        self.detail_root = train_root + '/detail/'
        

        self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts    = [self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith('.png')]
        self.ts     = [self.t_root + f for f in os.listdir(self.t_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.bodys  = [self.body_root + f for f in os.listdir(self.body_root) if f.endswith('.png')]
        self.details = [self.detail_root + f for f in os.listdir(self.detail_root) if f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts    = sorted(self.gts)
        self.ts     = sorted(self.ts)
        self.bodys  = sorted(self.bodys)
        self.details = sorted(self.details)

        # self.filter_files()
        self.size = len(self.images)

        ## RGB(VT5000 + VT1000 + VT821)
        # [0.525, 0.590, 0.537], [0.177, 0.167, 0.176]

        ## MIX(VT5000 + VT1000 + VT821)
        # [0.501, 0.612, 0.602], [0.173, 0.152, 0.166]


        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.525, 0.590, 0.537], [0.177, 0.167, 0.176])])


        ## T(VT5000 + VT1000 + VT821)
        # [0.736, 0.346, 0.339], [0.179, 0.196, 0.169]

        self.t_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.736, 0.346, 0.339], [0.179, 0.196, 0.169])])

        self.gt_transform     = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])
        self.body_transform   = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])
        self.detail_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        image  = self.rgb_loader(self.images[index])
        t      = self.rgb_loader(self.ts[index])
        gt     = self.binary_loader(self.gts[index])
        body   = self.binary_loader(self.bodys[index])
        detail = self.binary_loader(self.details[index])

        image,t,gt,body,detail = cv_random_flip(image,t,gt,body,detail)
        image,t,gt,body,detail = randomCrop(image,t,gt,body,detail )
        image,t,gt,body,detail = randomRotation(image,t,gt,body,detail )
        
        # image = randomGamma(image)
        image = colorEnhance(image)
        
        t     = colorEnhance(t)
        # gt=randomGaussian(gt)
        gt = randomPeper(gt)

        image  = self.img_transform(image)
        t      = self.t_transform(t)
        gt     = self.gt_transform(gt)
        body   = self.body_transform(body)
        detail = self.detail_transform(detail)
        
        return image, t, gt, body, detail

    '''def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts)==len(self.images)
        images = []
        gts = []
        depths=[]
        for img_path, gt_path,depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth= Image.open(depth_path)
            if img.size == gt.size and gt.size==depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths=depths'''

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    '''def resize(self, img,t,gt,body,detail):
        assert img.size == gt.size and gt.size==t.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR),t.resize((w, h), Image.BILINEAR),gt.resize((w, h), Image.NEAREST),body.resize((w, h), Image.NEAREST),detail.resize((w, h), Image.NEAREST)
        else:
            return img,t,gt,body,detail'''

    def __len__(self):
        return self.size

#dataloader for training
def get_loader(train_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=False):

    dataset = SalObjDataset(train_root,trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

#test dataset and loader
class test_dataset:
    def __init__(self, test_root, testsize):
        self.testsize = testsize

        self.image_root = test_root + '/RGB/'
        self.gt_root    = test_root + '/GT/'
        self.t_root     = test_root + '/T/'

        self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts    = [self.gt_root + f for f in os.listdir(self.gt_root) if f.endswith('.png') or f.endswith('.jpg')]
        self.ts     = [self.t_root + f for f in os.listdir(self.t_root) if f.endswith('.jpg') or f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts    = sorted(self.gts)
        self.ts     = sorted(self.ts)

        ## RGB(VT5000 + VT1000 + VT821)
        # [0.525, 0.590, 0.537], [0.177, 0.167, 0.176]
        ## MIX(VT5000 + VT1000 + VT821)
        # [0.501, 0.612, 0.602], [0.173, 0.152, 0.166]

        ## RGB(VT1606)
        # [0.238, 0.271, 0.236], [0.172, 0.174, 0.174]

        self.img_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.525, 0.590, 0.537], [0.177, 0.167, 0.176])])
        

        ## T(VT5000 + VT1000 + VT821)
        # [0.736, 0.346, 0.339], [0.179, 0.196, 0.169]
        ## T(VT251)
        # [0.273, 0.687, 0.716], [0.148, 0.212, 0.155]

        ## T(VT1606)
        # color
        # [0.213, 0.656, 0.779], [0.113, 0.203, 0.122]
        # gray
        # [0.3451, 0.345, 0.345], [0.0768, 0.0768, 0.0768]


        self.t_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.736, 0.346, 0.339], [0.179, 0.196, 0.169])])

        self.gt_transform = transforms.ToTensor()

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        shape = image.size
        image = self.img_transform(image).unsqueeze(0)

        t = self.rgb_loader(self.ts[self.index])
        t = self.t_transform(t).unsqueeze(0)

        gt    = self.binary_loader(self.gts[self.index])
        
        name = self.images[self.index].split('/')[-1]

        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        self.index = self.index % self.size
        return image, t, gt, shape, name
        # return image, t, shape, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    def __len__(self):
        return self.size