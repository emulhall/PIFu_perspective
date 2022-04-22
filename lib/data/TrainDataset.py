from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import glob
import logging
#import open3d as o3d
import matplotlib.pyplot as plt
import time
from ..geometry import *
#from ..mesh_util import save_obj_mesh_with_color

log = logging.getLogger('trimesh')
log.setLevel(40)

def load_trimesh(root_dir):
    folders = os.listdir(root_dir)
    meshs = {}
    max_point=-1
    min_point=1
    for i, f in enumerate(folders):
        sub_name = f.replace('.obj','')
        #if int(sub_name)==0:
        mesh = trimesh.load(os.path.join(root_dir, f),process=False,file_type='obj')
        meshs[sub_name] = mesh

    return meshs

def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


class TrainDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'perspective'
        self.projection = orthogonal if self.projection_mode== 'orthogonal' else perspective

        # Path setup
        self.root = self.opt.dataroot
        self.RENDER = os.path.join(self.root, '0','color')
        self.MASK = os.path.join(self.root, '0','gr_mask')
        self.PARAM = os.path.join(self.root, '0','camera')
        self.BBOX = os.path.join(self.root, '0','boundingBox')
        #self.UV_MASK = os.path.join(self.root, 'UV_MASK')
        #self.UV_NORMAL = os.path.join(self.root, 'UV_NORMAL')
        #self.UV_RENDER = os.path.join(self.root, 'UV_RENDER')
        #self.UV_POS = os.path.join(self.root, 'UV_POS')
        self.OBJ = os.path.join(self.root.replace('train_512_RenderPeople_all_sparse','meshes'))

        self.B_MIN = np.array([0, -1, -0.4])
        self.B_MAX = np.array([3.6, 3.0, 2.6])

        self.is_train = (phase == 'train')
        self.load_size = self.opt.loadSize

        self.num_views = self.opt.num_views

        self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_color = self.opt.num_sample_color

        #self.yaw_list = list(range(0,360,1))
        #self.pitch_list = [0]
        views = sorted(glob.glob(os.path.join(self.root,'*')))
        self.views=[]
        for v in views:
            _,num = os.path.split(v)
            if int(num)==56 or int(num)==83:
                continue
            else:
                self.views.append(num)



        self.subjects = self.get_subjects()

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])

        self.mesh_dic = load_trimesh(self.OBJ)

    def get_subjects(self):
        all_subjects = glob.glob(os.path.join(self.root,'0/color/*.png'))
        all_subjects = sorted([i.replace('.png','').replace(os.path.join(self.root,'0/color/'),'') for i in all_subjects])
        #all_subjects = ['%i'%i for i in range(0,1)]

        var_subjects = ['%i'%i for i in range(330,345)]
        var_subjects.append('76')
        #var_subjects=[]
        if len(var_subjects) == 0:
            return all_subjects

        if self.is_train:
            return sorted(list(set(all_subjects) - set(var_subjects)))
        else:
            return sorted(list(var_subjects))

    def __len__(self):
        return len(self.subjects) * len(self.views)

    def calculate_scaling_and_origin_rp(self,bbox, height):
        """
        Calculates the origin and scaling of an image for the RenderPeople dataset

        Parameters
        ----------
        bbox - the bounding box coords: ndarray of shape (4,2)
        height - the height of the final image: int
        """
        bb1_t=bbox-1
        bbc1_t=bb1_t[2:4,0:3]
        origin = np.multiply([bb1_t[1,0]-bbc1_t[1,0],bb1_t[0,0]-bbc1_t[0,0]],2)
        squareSize = np.maximum(bb1_t[0,1]-bb1_t[0,0]+1,bb1_t[1,1]-bb1_t[1,0]+1)
        scaling = np.multiply(np.true_divide(squareSize,height),2)

        return origin, scaling

    def get_render(self, subject, num_views, yid=0, pid=0, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''
        # The ids are an even distribution of num_views around view_id
        view_ids = [self.views[(yid + len(self.views) // num_views * offset) % len(self.views)] for offset in range(num_views)]

        if random_sample:
            view_ids = np.random.choice(self.views, num_views, replace=False)

        calib_list = []
        render_list = []
        mask_list = []
        extrinsic_list = []
        transforms_list = []

        for vid in view_ids:
            K_path = os.path.join(self.PARAM.replace('/0/','/%s/'%(vid)),'K.txt')
            R_path = os.path.join(self.PARAM.replace('/0/','/%s/'%(vid)),'R.txt')
            C_path = os.path.join(self.PARAM.replace('/0/','/%s/'%(vid)),'cen.txt')
            render_path = os.path.join(self.RENDER.replace('/0/','/%s/'%(vid)), subject+'.png')
            mask_path = os.path.join(self.MASK.replace('/0/','/%s/'%(vid)), subject+'.png')
            bbox_path = os.path.join(self.BBOX.replace('/0/','/%s/'%(vid)),subject+'.txt')

            # loading calibration data
            K = np.loadtxt(K_path,delimiter=',',dtype='f')
            R = np.loadtxt(R_path,delimiter=',',dtype='f')
            center = np.loadtxt(C_path,delimiter=',',dtype='f')
            bbox = np.loadtxt(bbox_path,delimiter=',',dtype='f')
            origin,scale = self.calculate_scaling_and_origin_rp(bbox,self.load_size)

            translate = -np.matmul(R, center).reshape(3, 1)
            extrinsic = np.concatenate([R, translate], axis=1)
            extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

            # Transform under image pixel space
            transforms_tensor=np.zeros((2,3))
            transforms_tensor[:1,:1]=(1 / scale)
            transforms_tensor[:2,2:3]=((1/scale)*origin.reshape(2,1))
            transforms_tensor = torch.Tensor(transforms_tensor)
            
            trans_intrinsic = np.identity(4)
            trans_intrinsic[:3,:3] = K

            mask = Image.open(mask_path).convert('L')
            render = Image.open(render_path).convert('RGB')

            #if self.is_train:
            #    render = self.aug_trans(render)

            intrinsic = trans_intrinsic
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            extrinsic = torch.Tensor(extrinsic).float()

            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float()
            mask = torch.where(mask>0,torch.ones_like(mask),torch.zeros_like(mask))
            mask_list.append(mask)

            render = self.to_tensor(render)
            render = mask.expand_as(render) * render

            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)
            transforms_list.append(transforms_tensor)

            verts_tensor = torch.Tensor(self.mesh_dic[subject].vertices).T
            faces = torch.Tensor(self.mesh_dic[subject].faces).T

            '''xyz_tensor = perspective(verts_tensor[np.newaxis,...], calib[np.newaxis,...], transforms_tensor[np.newaxis,...])
            uv = xyz_tensor[:, :2, :]
            color = index(render[np.newaxis,...], uv).detach().cpu().numpy()[0].T
            color = color * 0.5 + 0.5
            save_obj_mesh_with_color('/media/mulha024/i/PIFu-master/test_mesh.obj', verts_tensor.T, faces.T, color)
            print("Saved the mesh")'''

        return {
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'mask': torch.stack(mask_list, dim=0),
            'transforms': torch.stack(transforms_list,dim=0)
        }

    def select_sampling_method(self, subject, calibs, transforms):
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)
        
        mesh = self.mesh_dic[subject]
        vertices = torch.Tensor(mesh.vertices).T
        xyz = self.projection(vertices[np.newaxis,...].repeat(calibs.shape[0],1,1), calibs, transforms)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]
        in_img = torch.where((xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0))
        vertices = vertices[:,in_img[1]]
        rand_multiplier = 1
        surf_multiplier = 1
        
        if vertices.shape[1]>self.num_sample_inout:
            surf_multiplier = 6
            rand_multiplier = 1
        else:
            print(vertices.shape[1])
            print(subject)
            surf_multiplier = 1
            rand_multiview = 100
        surface_points = vertices[:,np.random.choice(np.arange(vertices.shape[1]), size=min(self.num_sample_inout*surf_multiplier,vertices.shape[1]), replace=False)]
        surface_points = surface_points.T

        #surface_points, _ = trimesh.sample.sample_surface(mesh, self.num_sample_inout*4)
        sample_points = surface_points + np.random.normal(scale=self.opt.sigma, size=surface_points.shape)
        # add random points within image space
        length = self.B_MAX - self.B_MIN

        random_points = np.random.rand(self.num_sample_inout*rand_multiplier, 3) * length + self.B_MIN
        xyz = self.projection(torch.Tensor(random_points[np.newaxis,...]).permute(0,2,1).repeat(calibs.shape[0], 1, 1), calibs, transforms)
        xy = xyz[:, :2, :]
        z = xyz[:, 2:3, :]
        in_img = torch.where((xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0))
        random_points = random_points[in_img[1],:]

        sample_points = np.concatenate([sample_points, random_points], 0)
        np.random.shuffle(sample_points)

        inside = mesh.contains(sample_points)
        inside_points = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        nin = inside_points.shape[0]
        inside_points = inside_points[
                        :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
        outside_points = outside_points[
                         :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points[
                                                                                               :(self.num_sample_inout - nin)]
        #print("Inside to outside ratio:")
        #print(inside_points.shape[0]/outside_points.shape[0])


        samples = np.concatenate([inside_points, outside_points], 0).T
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)

        #save_samples_truncted_prob('out.ply', samples.T, labels.T)
        # exit()

        if samples.shape[1]!=self.num_sample_inout:
            print(vertices.shape)
            print(random_points.shape)
            print(subject)

        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()

        #save_samples_truncted_prob('test_samples.ply', samples, labels)
        
        del mesh

        return {
            'samples': samples,
            'labels': labels
        }


    def get_color_sampling(self, subject, yid, pid=0):
        yaw = self.yaw_list[yid]
        pitch = self.pitch_list[pid]
        uv_render_path = os.path.join(self.UV_RENDER, subject, '%d_%d_%02d.jpg' % (yaw, pitch, 0))
        uv_mask_path = os.path.join(self.UV_MASK, subject, '%02d.png' % (0))
        uv_pos_path = os.path.join(self.UV_POS, subject, '%02d.exr' % (0))
        uv_normal_path = os.path.join(self.UV_NORMAL, subject, '%02d.png' % (0))

        # Segmentation mask for the uv render.
        # [H, W] bool
        uv_mask = cv2.imread(uv_mask_path)
        uv_mask = uv_mask[:, :, 0] != 0
        # UV render. each pixel is the color of the point.
        # [H, W, 3] 0 ~ 1 float
        uv_render = cv2.imread(uv_render_path)
        uv_render = cv2.cvtColor(uv_render, cv2.COLOR_BGR2RGB) / 255.0

        # Normal render. each pixel is the surface normal of the point.
        # [H, W, 3] -1 ~ 1 float
        uv_normal = cv2.imread(uv_normal_path)
        uv_normal = cv2.cvtColor(uv_normal, cv2.COLOR_BGR2RGB) / 255.0
        uv_normal = 2.0 * uv_normal - 1.0
        # Position render. each pixel is the xyz coordinates of the point
        uv_pos = cv2.imread(uv_pos_path, 2 | 4)[:, :, ::-1]

        ### In these few lines we flattern the masks, positions, and normals
        uv_mask = uv_mask.reshape((-1))
        uv_pos = uv_pos.reshape((-1, 3))
        uv_render = uv_render.reshape((-1, 3))
        uv_normal = uv_normal.reshape((-1, 3))

        surface_points = uv_pos[uv_mask]
        surface_colors = uv_render[uv_mask]
        surface_normal = uv_normal[uv_mask]

        if self.num_sample_color:
            sample_list = random.sample(range(0, surface_points.shape[0] - 1), self.num_sample_color)
            surface_points = surface_points[sample_list].T
            surface_colors = surface_colors[sample_list].T
            surface_normal = surface_normal[sample_list].T

        # Samples are around the true surface with an offset
        normal = torch.Tensor(surface_normal).float()
        samples = torch.Tensor(surface_points).float() \
                  + torch.normal(mean=torch.zeros((1, normal.size(1))), std=self.opt.sigma).expand_as(normal) * normal

        # Normalized to [-1, 1]
        rgbs_color = 2.0 * torch.Tensor(surface_colors).float() - 1.0

        return {
            'color_samples': samples,
            'rgbs': rgbs_color
        }

    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        # try:
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)
        yid = tmp % len(self.views)
        pid = tmp // len(self.views)

        # name of the subject 'rp_xxxx_xxx'
        subject = self.subjects[sid]
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.OBJ, subject + '.obj'),
            'sid': sid,
            'yid': yid,
            'pid': pid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }
        render_data = self.get_render(subject, num_views=self.num_views, yid=yid, pid=pid,
                                        random_sample=self.opt.random_multiview)
        res.update(render_data)

        if self.opt.num_sample_inout:
            sample_data = self.select_sampling_method(subject, res['calib'], res['transforms'])
            res.update(sample_data)
        
        # img = np.uint8((np.transpose(render_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
        # rot = render_data['calib'][0,:3, :3]
        # trans = render_data['calib'][0,:3, 3:4]
        # pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] > 0.5])  # [3, N]
        # pts = 0.5 * (pts.numpy().T + 1.0) * render_data['img'].size(2)
        # for p in pts:
        #     img = cv2.circle(img, (p[0], p[1]), 2, (0,255,0), -1)
        # cv2.imshow('test', img)
        # cv2.waitKey(1)

        if self.num_sample_color:
            color_data = self.get_color_sampling(subject, yid=yid, pid=pid)
            res.update(color_data)
        return res
        # except Exception as e:
        #     print(e)
        #     return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)