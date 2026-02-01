import torch
from torch.utils.data import Dataset, DataLoader
import random, os, json, cv2
from glob import glob
# import matplotlib.pyplot as plt
from .homo_utils import generate_homo, regenerate_homo_cv2, use_homo
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
from natsort import natsorted

def get_coord_offsets(pts_path, img_a_orig_size, img_b_orig_size):
    """
    计算在 320x320 空间下，指定点在 A 和 B 之间的坐标偏移量
    :param pts_path: txt标签路径 (x1 y1 x2 y2)
    :param img_a_orig_size: 图像A原始尺寸 (宽, 高)
    :param img_b_orig_size: 图像B原始尺寸 (宽, 高)
    """
    target_size = 320
    
    # 1. 读取并缩放坐标点到 320x320 空间
    pts_a, pts_b = [], []
    scale_a = (target_size / img_a_orig_size[0], target_size / img_a_orig_size[1])
    scale_b = (target_size / img_b_orig_size[0], target_size / img_b_orig_size[1])
    
    try:
        with open(pts_path, 'r') as f:
            for line in f:
                c = list(map(float, line.split()))
                if len(c) == 4:
                    # 映射到 320x320 坐标系
                    pts_a.append([c[0] * scale_a[0], c[1] * scale_a[1]])
                    pts_b.append([c[2] * scale_b[0], c[3] * scale_b[1]])
    except Exception as e:
        return f"读取文件失败: {e}"

    pts_a = np.array(pts_a, dtype=np.float32)
    pts_b = np.array(pts_b, dtype=np.float32)

    if len(pts_a) < 4:
        return "对应点数量不足，无法计算单应性矩阵。"

    # 2. 计算从 B 到 A 的单应性矩阵 H
    H, _ = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 5.0)
    
    # 3. 计算 H 的逆矩阵 (从 A 映射到 B)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return "矩阵奇异，无法求逆，请检查坐标点是否共线。"

    # 4. 定义 A 图像中的四个测试点
    test_pts_a = np.array([
        [32, 32],
        [288, 32],
        [32, 288],
        [288, 288]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # 5. 变换这些点到 B 的坐标系
    test_pts_b = cv2.perspectiveTransform(test_pts_a, H_inv).reshape(-1, 2)
    test_pts_a = test_pts_a.reshape(-1, 2)

    # 6. 计算偏移量 (B坐标 - A坐标)
    results = {}
    print(f"{'点 (A)':<15} {'对应点 (B)':<20} {'偏移 (dx, dy)':<20}")
    print("-" * 55)
    
    for i in range(len(test_pts_a)):
        pa = test_pts_a[i]
        pb = test_pts_b[i]
        dx = pb[0] - pa[0]
        dy = pb[1] - pa[1]
        
        point_key = tuple(pa.astype(int))
        results[point_key] = [dx, dy]
        
        print(f"{str(point_key):<15} ({pb[0]:.2f}, {pb[1]:.2f})    ({dx:+.2f}, {dy:+.2f})")

    return results

def center_crop(img, target_size=256):
    """执行中心裁剪"""
    h, w = img.shape[:2]
    start_y = (h - target_size) // 2
    start_x = (w - target_size) // 2
    return img[start_y:start_y+target_size, start_x:start_x+target_size], start_x, start_y

def crop_image_random(image1, image2, width, height, random_int=None, save_window=False):
    """
    随机位置裁剪指定大小的图像块
    :param image_path: 输入图像路径
    :param width: 裁剪宽度
    :param height: 裁剪高度
    :return: 裁剪后的图像数组
    """
    if image1 is None:
        raise FileNotFoundError("无法加载图像")
    
    img_height, img_width = image1.shape[:2]
    
    # 计算随机偏移量
    max_x = img_width - width
    max_y = img_height - height
    if max_x < 0 or max_y < 0:
        raise ValueError("裁剪尺寸超过图像尺寸")
    
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    if random_int:
        x, y = random_int[0], random_int[1]

    # 执行裁剪
    cropped1 = image1[y:y+height, x:x+width]
    cropped2 = image2[y:y+height, x:x+width]
    if save_window:
        return cropped1, cropped2, x, y
    else:
        return cropped1, cropped2
    
def crop_image_random_three(image1, image2, image_truth, width, height, random_int=None, save_window=False):
    """
    随机位置裁剪指定大小的图像块
    :param image_path: 输入图像路径
    :param width: 裁剪宽度
    :param height: 裁剪高度
    :return: 裁剪后的图像数组
    """
    if image1 is None:
        raise FileNotFoundError("无法加载图像")
    
    img_height, img_width = image1.shape[:2]
    
    # 计算随机偏移量
    max_x = img_width - width
    max_y = img_height - height
    if max_x < 0 or max_y < 0:
        raise ValueError("裁剪尺寸超过图像尺寸")
    
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    if random_int:
        x, y = random_int[0], random_int[1]

    # 执行裁剪
    cropped1 = image1[y:y+height, x:x+width]
    cropped2 = image2[y:y+height, x:x+width]
    cropped_truth = image_truth[y:y+height, x:x+width]
    #test resolution
    # cropped1 = image1[0:512, 0:512]
    # cropped2 = image2[0:512, 0:512]
    # cropped_truth = image_truth[0:512, 0:512]
    if save_window:
        return cropped1, cropped2, x, y
    else:
        return cropped1, cropped2, cropped_truth


class homo_dataset(Dataset):
    def __init__(self, split, dataset, args):
        self.dataset = dataset
        self.args = args
        self.split = split
        self.homo_parameter = {"marginal":96, "perturb":96, "patch_size":256}
        if args.diffusion_phase == 'homo2':
            self.homo_parameter = {"marginal":32, "perturb":16, "patch_size":256}
        elif args.diffusion_phase == 'homo3':
            self.homo_parameter = {"marginal":32, "perturb":8, "patch_size":256}
        self.dataset_name = dataset
        self.split = split

        if split == 'train':
            if dataset =='mscoco':
                root_img2 = '/data/data0/zhk/mscoco/train2017'
                root_img1 = '/data/data0/zhk/mscoco/train2017'
            if dataset == 'ggmap':
                root_img2 = '/data/data0/zhk/GoogleMap/train2014_input'
                root_img1 = '/data/data0/zhk/GoogleMap/train2014_template_original'  
            if dataset == 'spid':
                root_img1 = '/home/csy/datasets/csy/moving_object/img_pair_train_new/img1'
                root_img2 = '/home/csy/datasets/csy/moving_object/img_pair_train_new/img2'                
            if dataset == 'rgb_nir':
                root_img1 = '/mnt/hdd1/gyh/Dataset/HomoDiff/rgb_nir/train/nir'
                root_img2 = '/mnt/hdd1/gyh/Dataset/HomoDiff/rgb_nir/train/rgb'
            if dataset == 'flir':
                root_img1 = '/mnt/hdd1/gyh/Dataset/HomoDiff/flir/train/ir'
                root_img2 = '/mnt/hdd1/gyh/Dataset/HomoDiff/flir/train/rgb'
            if dataset == 'hypersim':
                root_img1 = '/mnt/hdd1/gyh/Dataset/HomoDiff/hypersim/train/depth'
                root_img2 = '/mnt/hdd1/gyh/Dataset/HomoDiff/hypersim/train/rgb'
        elif split == 'val':
            if dataset=='mscoco':
                root_img2 = '/data/data0/zhk/mscoco/test2017'
                root_img1 = '/data/data0/zhk/mscoco/test2017'
            if dataset == 'ggmap':
                root_img2 = '/data/data0/zhk/GoogleMap/val2014_input'        
                root_img1 = '/data/data0/zhk/GoogleMap/val2014_template_original'
            if dataset == 'spid':
                root_img1 = '/home/csy/datasets/csy/moving_object/img_pair_test_new/img1'
                root_img2 = '/home/csy/datasets/csy/moving_object/img_pair_test_new/img2'
            if dataset == 'rgb_nir':
                root_img1 = '/mnt/hdd1/gyh/Dataset/HomoDiff/rgb_nir/val/nir'
                root_img2 = '/mnt/hdd1/gyh/Dataset/HomoDiff/rgb_nir/val/rgb'
            if dataset == 'flir':
                root_img1 = '/mnt/hdd1/gyh/Dataset/HomoDiff/flir/val/ir'
                root_img2 = '/mnt/hdd1/gyh/Dataset/HomoDiff/flir/val/rgb'
            if dataset == 'hypersim':
                root_img1 = '/mnt/hdd1/gyh/Dataset/HomoDiff/hypersim/val/depth'
                root_img2 = '/mnt/hdd1/gyh/Dataset/HomoDiff/hypersim/val/rgb'
        elif split == 'test':
            if dataset=='mscoco':
                root_img2 = '/data/data0/zhk/mscoco/test2017'
                root_img1 = '/data/data0/zhk/mscoco/test2017'
            if dataset == 'ggmap':
                root_img2 = '/data/data0/zhk/GoogleMap/val2014_input'        
                root_img1 = '/data/data0/zhk/GoogleMap/val2014_template_original'
            if dataset == 'spid':
                root_img1 = '/home/csy/datasets/csy/moving_object/img_pair_test_new/img1'
                root_img2 = '/home/csy/datasets/csy/moving_object/img_pair_test_new/img2'
            if dataset == 'rgb_nir':
                root_img1 = '/mnt/hdd1/gyh/Dataset/HomoDiff/rgb_nir/test/nir'
                root_img2 = '/mnt/hdd1/gyh/Dataset/HomoDiff/rgb_nir/test/rgb'
                root_H_s2t = '/mnt/hdd1/gyh/Dataset/HomoDiff/flir/test/H_s2t'
            if dataset == 'flir':
                root_img1 = '/mnt/hdd1/gyh/Dataset/HomoDiff/flir/val/ir'
                root_img2 = '/mnt/hdd1/gyh/Dataset/HomoDiff/flir/val/rgb'
                # root_img2 = '/mnt/hdd1/gyh/Dataset/HomoDiff/flir/denoise_ADFNet'
                root_H_s2t = '/mnt/hdd1/gyh/Dataset/HomoDiff/flir/test/H_s2t'
            if dataset == 'hypersim':
                root_img1 = '/mnt/hdd1/gyh/Dataset/HomoDiff/hypersim/test/depth'
                root_img2 = '/mnt/hdd1/gyh/Dataset/HomoDiff/hypersim/test/rgb'
                root_H_s2t = '/mnt/hdd1/gyh/Dataset/HomoDiff/hypersim/test/H_s2t'

        if split == 'single_pair':
            # self.image_list_img1 = ['/mnt/hdd1/gyh/Dataset/multispectraldata/RGB_NIR/myrgbnir/img1.png']
            # self.image_list_img2 = ['/mnt/hdd1/gyh/Dataset/multispectraldata/RGB_NIR/myrgbnir/img2.png']
            # self.image_list_img1 = ['/mnt/hdd1/gyh/Dataset/multispectraldata/RGB_NIR/lion/img1.png']
            # self.image_list_img2 = ['/mnt/hdd1/gyh/Dataset/multispectraldata/RGB_NIR/lion/img2.png']
            self.image_list_img1 = ['/mnt/hdd1/gyh/Dataset/multispectraldata/RGB_NIR/orchid/img1.png']
            self.image_list_img2 = ['/mnt/hdd1/gyh/Dataset/multispectraldata/RGB_NIR/orchid/img2.png']
        else:
            self.image_list_img1 = natsorted(glob(os.path.join(root_img1, '*.jpg')))
            self.image_list_img2 = natsorted(glob(os.path.join(root_img2, '*.jpg')))

        if split == 'test':
            self.H_s2t_list = natsorted(glob(os.path.join(root_H_s2t, '*.json')))
        if len(self.image_list_img1) == 0:
            self.image_list_img1 = natsorted(glob(os.path.join(root_img1, '*.png')))
        if len(self.image_list_img2) == 0:  
            self.image_list_img2 = natsorted(glob(os.path.join(root_img2, '*.png')))            

        self.sigma = args.noise_sigma


    def __len__(self):
        return len(self.image_list_img2)

    def __getitem__(self, index):
        if self.split == 'test':
            image_name = self.image_list_img2[index]
            json_name = self.H_s2t_list[index]
        if self.dataset_name == 'hypersim':
            img2 = cv2.imread(self.image_list_img2[index])
            # img2 = cv2.imread(self.image_list_img2[index])
            img_16bit = cv2.imread(self.image_list_img1[index], cv2.IMREAD_ANYDEPTH)
            img_8bit = cv2.normalize(img_16bit, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            img1 = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)
            scale_factor = 0.5
            new_size = (int(img1.shape[1] * scale_factor), int(img1.shape[0] * scale_factor))
            img1 = cv2.resize(
                img1, 
                new_size, 
                interpolation=cv2.INTER_AREA  # 缩小时推荐使用区域插值
            )
            img2 = cv2.resize(
                img2, 
                new_size, 
                interpolation=cv2.INTER_AREA  # 缩小时推荐使用区域插值
            )
        else:
            # if self.split != 'train':
            #     print('index', index)
            #     print('len(self.image_list_img1)', len(self.image_list_img1))
            img1 = cv2.imread(self.image_list_img1[index])
            img2 = cv2.imread(self.image_list_img2[index])
            # img1 = cv2.imread(self.image_list_img1[index])
            # img2 = cv2.imread(self.image_list_img2[index])
            # scale_factor = 0.7
            # new_size = (int(img1.shape[1] * scale_factor), int(img1.shape[0] * scale_factor))
            # img1 = cv2.resize(
            #     img1, 
            #     new_size, 
            #     interpolation=cv2.INTER_AREA  # 缩小时推荐使用区域插值
            # )
            # img2 = cv2.resize(
            #     img2, 
            #     new_size, 
            #     interpolation=cv2.INTER_AREA)
            # print('img1.shape', img1.shape)

        if self.split == 'test':
            with open(self.H_s2t_list[index], 'r') as json_file:
                data = json.load(json_file)
            # H_s2t = torch.tensor(data['H']).float()
            window = data['window']
            org_pts = np.array(data['org_pts'], dtype=np.float32)
            dst_pts = np.array(data['dst_pts'], dtype=np.float32)
        # gauss = np.random.normal(0, self.sigma, img1.shape)
        # gauss = np.random.normal(0, 0, img1.shape)
        # noisy = img1.astype(np.float32) + gauss.astype(np.float32)
        # img1_truth = img1.copy()
        # img1 = np.clip(noisy, 0, 255).astype(np.uint8)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        if self.split == 'single_pair':
            img1_org = img1.copy()
            img2_org = img2.copy()
        # img1_truth = cv2.cvtColor(img1_truth, cv2.COLOR_BGR2RGB)
        # img_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)

        # img1, img2 = crop_image_random(noisy, img2, 256, 256)
        # img1, img2 = crop_image_random(noisy, img2, 320, 320)
        if self.split != 'test':
            window = torch.tensor([0, 0])

        if self.split != 'test':
            img1, img2 = crop_image_random(img1, img2, self.homo_parameter['patch_size'] + 2*self.homo_parameter['marginal'], self.homo_parameter['patch_size'] + 2*self.homo_parameter['marginal'])
        else:
            img1, img2  = crop_image_random(img1, img2, self.homo_parameter['patch_size'] + 2*self.homo_parameter['marginal'], self.homo_parameter['patch_size'] + 2*self.homo_parameter['marginal'], random_int=window)
        if self.dataset == 'spid':
            img_size = self.homo_parameter["patch_size"] + 2 * self.homo_parameter["marginal"]
            img1 = cv2.resize(img1, (img_size, img_size))
            img2 = cv2.resize(img2, (img_size, img_size))
        elif self.dataset == "mscoco":
            img1 = cv2.resize(img1, (320, 240))
            img2 = cv2.resize(img2, (320, 240))

        self.homo_parameter["height"], self.homo_parameter["width"], _ = img1.shape

        if self.split == 'test':
            # corners = np.array(
            #     [[0, 0, 1], [self.homo_parameter["width"]-1, 0, 1], [0, self.homo_parameter["height"]-1, 1], [self.homo_parameter["width"] - 1, self.homo_parameter["height"] - 1, 1]]
            # )
            # real_warped_corners = np.dot(corners, np.transpose(H_s2t))
            # real_warped_corners = (
            #     real_warped_corners[:, :2] / real_warped_corners[:, 2:]
            # )
            # print('corners', corners)
            # print('real_warped_corners', real_warped_corners)
            # exit()
            patch_img1_warp, patch_img2, four_gt, org_pts, dst_pts, large_img1_warp, large_img2 = use_homo(img1, img2, homo_parameter=self.homo_parameter, org_pts=org_pts, dst_pts=dst_pts)
            # print('real_warped_corners.shape', real_warped_corners.shape)
            # exit()
        else:
            patch_img1_warp, patch_img2, four_gt, org_pts, dst_pts, large_img1_warp, large_img2 = generate_homo(img1, img2, homo_parameter=self.homo_parameter, transform=None)

            # print('four_gt.shape', four_gt.shape)
            # exit()
        noise = torch.randn(patch_img1_warp.size())
        noise = noise/torch.std(noise,unbiased=False)
        noise=noise.mul_(self.sigma/255.0)
        img_noise = patch_img2.clone().detach()
        img_noise.add_(noise)
        if self.split == 'single_pair':
            img1_320 = cv2.resize(img1_org, (320, 320))
            img2_320 = cv2.resize(img2_org, (320, 320))
            img_noise = torch.from_numpy(center_crop(img1_320, 256)[0]).float().permute(2, 0, 1)/255.
            patch_img1_warp = torch.from_numpy(center_crop(img2_320, 256)[0]).float().permute(2, 0, 1)/255.
            # print('img_noise', img_noise)
            # print('patch_img1_warp', patch_img1_warp)
            # exit()
        # img_noise = torch.clamp(img_noise, 0 ,1)
        if self.split == 'val':
            return {"noisy":2*img_noise-1, "clean":2*patch_img1_warp-1, "flow_truth":four_gt, "truth":2*patch_img2-1,
                "org_pts":org_pts, "dst_pts":dst_pts,
                "large_img1_warp":large_img1_warp, "large_img2":large_img2}
        elif self.split == 'test':
            return {"noisy":2*img_noise-1, "clean":2*patch_img1_warp-1, "flow_truth":four_gt, "truth":2*patch_img2-1,
                    "org_pts":org_pts, "dst_pts":dst_pts,
                    "large_img1_warp":large_img1_warp, "large_img2":large_img2, "image_name":image_name, "json_name":json_name}
        elif self.split == 'train':
            return {"noisy":2*img_noise-1, "clean":2*patch_img1_warp-1, "flow_truth":four_gt, "truth":2*patch_img2-1,
                    "org_pts":org_pts, "dst_pts":dst_pts,
                    "large_img1_warp":large_img1_warp, "large_img2":large_img2}        
        elif self.split == 'single_pair':
            return {"noisy":2*img_noise-1, "clean":2*patch_img1_warp-1, "flow_truth":four_gt, "truth":2*patch_img2-1,
                    "org_pts":org_pts, "dst_pts":dst_pts,
                    "large_img1_warp":img1_320, "large_img2":img2_320}    

def fetch_dataloader(args, split='train'):
    if split == 'train':
        if args.dataset == "googleearth": dataset = GoogleEarth(split='train')
        else: dataset = homo_dataset(split='train', dataset=args.dataset, args=args)
        # dataset = Subset(dataset, indices=list(range(0, 350)))
        if args.dataset == 'flir':
            print('len(dataset)', len(dataset))
            print('flir subset')
        if args.dataset == 'hypersim':
            print('hypersim subset')            
            # test_indices = list(range(0, len(dataset), 10))
            # train_indices = [idx for idx in range(len(dataset)) if idx not in test_indices]
            # dataset = Subset(dataset, indices=train_indices)

        dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=16, drop_last=False)
        print('Training with %d image pairs' % len(dataset))
    elif split == 'test':
        if args.dataset == "googleearth": dataset = GoogleEarth(split='test')
        else: dataset = homo_dataset(split='test', dataset=args.dataset, args=args)
        # dataset = Subset(dataset, indices=list(range(0, 350)))
        if args.dataset == 'flir':
            print('flir dataset')  
            dataset = Subset(dataset, indices=list(range(403)))
        if args.dataset == 'hypersim':
            print('hypersim subset')            
            # test_indices = list(range(0, len(dataset), 10))
            # dataset = Subset(dataset, indices=test_indices)
            # dataset = Subset(dataset, indices=list(range(1084)))
        dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=16, drop_last=False)
        print('Training with %d image pairs' % len(dataset))
    elif split == 'single_pair':
        dataset = homo_dataset(split='single_pair', dataset=args.dataset, args=args)
        dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=16, drop_last=False)
    else: 
        dataset = homo_dataset(split='val', dataset=args.dataset, args=args)
        # dataset = Subset(dataset, indices=list(range(350, 377)))
        if args.dataset == 'flir':
            print('flir subset')
            # dataset = Subset(dataset, indices=list(range(3500, 4000)))
        if args.dataset == 'hypersim':
            print('hypersim subset')
            test_indices = list(range(0, len(dataset), 10))
            dataset = Subset(dataset, indices=test_indices)
        dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=16, drop_last=False)
        print('Validate with %d image pairs' % len(dataset))
    return dataloader

