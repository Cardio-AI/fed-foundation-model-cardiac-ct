from copy import deepcopy
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from kornia.geometry import transform as kgt
import monai.transforms as T
from monai.data.meta_tensor import MetaTensor
import torchio as tio
import raster_geometry as rg
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ImageNormalization
from src.data.utils import resample_image

INTENSITYPROPERTIES = {
    "max": 2176.0,
    "mean": 373.7686767578125,
    "median": 352.0,
    "min": -813.0,
    "percentile_00_5": -42.0,
    "percentile_99_5": 878.0,
    "std": 166.8054656982422
}

def torch_to_sitk(tensor, ref_sitk):
    # arr = np.einsum('xyz->zyx', tensor.detach().cpu().numpy().astype(np.float32))
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(ref_sitk)
    return img

def sitk_to_torch(img_sitk):
    arr = sitk.GetArrayFromImage(img_sitk)
    # arr = np.einsum('zyx->xyz', arr)
    return torch.from_numpy(arr.astype(np.float32))[None]

def np_to_sitk(arr, ref_sitk):
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(ref_sitk)
    return img

def draw_spheres_from_physical_points(img_sitk, points, radius=4, onehot=True):
    x1, y1, z1 = img_sitk.GetSize()
    hps_img = np.zeros((len(points), z1, y1, x1)) if onehot else np.zeros((z1, y1, x1))

    sphere = rg.sphere(2*radius, radius)
    # sphere = self.create_sphere(radius).astype(int)
    for c, pt in enumerate(points):
        xidx, yidx, zidx = img_sitk.TransformPhysicalPointToIndex(pt)
        if xidx >= x1 or yidx >= y1 or zidx >= z1:
            continue
        if zidx < 0:
            zidx = z1 + zidx
        xidx1, xidx2 = max(xidx-radius, 0), min(xidx+radius, x1)
        yidx1, yidx2 = max(yidx-radius, 0), min(yidx+radius, y1)
        zidx1, zidx2 = max(zidx-radius, 0), min(zidx+radius, z1)
        _sphere = deepcopy(sphere)
        if xidx1 == 0:
            _sphere = _sphere[-zidx2:]
        if xidx2 == x1:
            _sphere = _sphere[:(zidx2 - zidx1)]
        if yidx1 == 0:
            _sphere = _sphere[:,-yidx2:]
        if yidx2 == x1:
            _sphere = _sphere[:,:(yidx2 - yidx1)]
        if zidx1 == 0:
            _sphere = _sphere[:,:,-xidx2:]
        if zidx2 == x1:
            _sphere = _sphere[:,:,:(xidx2 - xidx1)]
        try:
            if onehot:
                hps_img[c,zidx1:zidx2,yidx1:yidx2,xidx1:xidx2] = _sphere
            else:
                hps_img[zidx1:zidx2,yidx1:yidx2,xidx1:xidx2] = _sphere * (c + 1)
        except:
            pass
    return hps_img

class CTNormalization(ImageNormalization):
    leaves_pixels_outside_mask_at_zero_if_use_mask_for_norm_is_true = False

    @staticmethod
    def type_to_clip_fn(img):
        if type(img) == np.ndarray:
            return np.clip
        elif type(img) == torch.Tensor or type(img) == MetaTensor:
            return torch.clamp
        elif type(img) == sitk.Image:
            return lambda x, lb, ub: sitk.Clamp(x, lowerBound=lb, upperBound=ub)

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensityproperties is not None, "CTNormalization requires intensity properties"
        # image = image.float()
        mean_intensity = self.intensityproperties['mean']
        std_intensity = self.intensityproperties['std']
        lower_bound = self.intensityproperties['percentile_00_5']
        upper_bound = self.intensityproperties['percentile_99_5']
        # clip_fn = np.clip if type(image) == np.ndarray else torch.clamp
        clip_fn = self.type_to_clip_fn(image)
        image = clip_fn(image, lower_bound, upper_bound)
        image = (image - mean_intensity) / max(std_intensity, 1e-8)
        return image

def crop_or_pad_img(img, target_shape=(192,192,192)):
    t = tio.transforms.CropOrPad(target_shape)
    img_tio = tio.ScalarImage.from_sitk(img)
    # seg = tio.LabelMap.from_sitk(seg)
    return t(img_tio).as_sitk() #, t(seg).as_sitk()

def crop_or_pad_seg(seg, target_shape=(192,192,192)):
    t = tio.transforms.CropOrPad(target_shape)
    seg_tio = tio.LabelMap.from_sitk(seg)
    return t(seg_tio).as_sitk()

def adjust_contast(img, seg=None, std_beta=0.05):
    beta = np.random.normal(loc=0., scale=std_beta)
    t = T.AdjustContrast(gamma=np.exp(beta))
    if seg is not None:
        img[seg > 0] = t(img)[seg > 0]
    else:
        img = t(img)
    return img

def rotate(img, seg=None, std_rot_z_deg=5.):
    rot_z_rad = np.random.normal(loc=0., scale=std_rot_z_deg) / 360 * 2. * np.pi
    ti = T.Rotate(angle=(0,0,rot_z_rad),mode='bilinear')
    img = ti(img)
    if seg is not None:
        ts = T.Rotate(angle=(0,0,rot_z_rad),mode='nearest')
        seg = ts(seg)
    return img, seg

def inject_noise(img, scale_upper_bound=0.03):
    scale = np.random.uniform(low=0., high=scale_upper_bound)
    noise = torch.randn_like(img) * scale
    return img + noise

def simulate_motion(img, translation=2):
    t = tio.transforms.RandomMotion(degrees=0, translation=translation, image_interpolation='bspline', num_transforms=1)
    img = tio.ScalarImage.from_sitk(img)
    return t(img).as_sitk()

def keep_shape(x, fn):
    in_shp = x.shape
    if len(in_shp) == 4:
        x = x[None]
    xt = fn(x)
    if len(in_shp) == 4:
        xt = xt[0]
    return xt

class RotateTorch:
    def __init__(self, rot_x=5, rot_y=5, rot_z=5, p=0.2, random=True):
        self.angles = np.radians([rot_x, rot_y, rot_z])
        self.random = random
        self.p = p
    
    def __call__(self, img, label, return_inverse=True):
        if np.random.uniform() > self.p:
            if return_inverse:
                return img, label, self.__class__(0, 0, 0, p=-1, random=False)
            return img, label
        
        if self.random:
            angles = np.random.normal(size=(3,)) * self.angles
        else:
            angles = self.angles
        translations = torch.zeros(3).to(img.device) # torch.tensor(img.size())[-3:] / 2
        R = self.transformation_matrix(angles, translations).to(img.device)

        rotated_img = kgt.warp_affine3d(img[None], M=R[None,:3], dsize=img.shape[-3:])[0]
        rotated_label = kgt.warp_affine3d(label[None], M=R[None,:3], dsize=img.shape[-3:], flags='nearest')[0]
        
        if return_inverse:
            return rotated_img, rotated_label, self.__class__(*-np.rad2deg(angles), p=1.1, random=False)
        return rotated_img, rotated_label
    
    @staticmethod
    def transformation_matrix(angles, translations):
        angle_x, angle_y, angle_z = angles
        Rx = torch.tensor([[1, 0, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x), 0],
                    [0, np.sin(angle_x), np.cos(angle_x), 0],
                    [0, 0, 0, 1]])

        Ry = torch.tensor([[np.cos(angle_y), 0, np.sin(angle_y), 0],
                        [0, 1, 0, 0],
                        [-np.sin(angle_y), 0, np.cos(angle_y), 0],
                        [0, 0, 0, 1]])

        Rz = torch.tensor([[np.cos(angle_z), -np.sin(angle_z), 0, 0],
                        [np.sin(angle_z), np.cos(angle_z), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        R = torch.mm(torch.mm(Rx, Ry), Rz)
        R[:3,3] = translations
        return R.float()
        

class ScaleTorch:
    def __init__(self, scale, p=0.2, random=True):
        self.scale = scale
        self.random = random
        self.p = p
    
    def __call__(self, img, label, return_inverse=True):
        if np.random.uniform() > self.p:
            if return_inverse:
                return img, label, self.__class__(scale=1, p=-1, random=False)
            return img, label
        
        if self.random:
            scale_factor = np.random.uniform(*self.scale)
        else:
            scale_factor = self.scale
        
        scaled_img = F.interpolate(img[None], scale_factor=scale_factor, mode='trilinear', align_corners=True)[0]
        scaled_label = F.interpolate(label[None], scale_factor=scale_factor, mode='nearest')[0]
        if scale_factor < 1:
            pad = np.array(img.size()) - np.array(scaled_img.size())
            pad = np.array([[p//2, p - p//2] for p in pad[-3:]]).flatten()
            scaled_img = F.pad(scaled_img, tuple(pad), 'constant', value=0)
            scaled_label = F.pad(scaled_label, tuple(pad), 'constant', value=0)
        elif scale_factor > 1:
            scaled_img = self.center_crop(scaled_img, img.size()[-3:])
            scaled_label = self.center_crop(scaled_label, label.size()[-3:])
        if return_inverse:
            inverse_scale = 1 / scale_factor
            return scaled_img, scaled_label, self.__class__(scale=inverse_scale, p=1.1, random=False)
        return scaled_img, scaled_label
    
    @staticmethod
    def center_crop(x, new_size):
        c, d, h, w = x.shape
        new_d, new_h, new_w = new_size

        start_d = max((d - new_d) // 2, 0)
        start_h = max((h - new_h) // 2, 0)
        start_w = max((w - new_w) // 2, 0)

        end_d = start_d + new_d
        end_h = start_h + new_h
        end_w = start_w + new_w

        cropped_x = x[:, start_d:end_d, start_h:end_h, start_w:end_w]
        return cropped_x


class GaussianBlurTorch:
    def __init__(self, sigma_range, kernel_size=5, p=0.2):
        self.sigma_range = sigma_range
        self.kernel_size = kernel_size
        self.p = p
    
    def __call__(self, img):
        if np.random.uniform() < self.p:
            sigma = np.random.uniform(*self.sigma_range)
            kernel = self.gaussian_kernel(self.kernel_size, sigma).to(img.device)
            img = F.conv3d(img[None], kernel[None,None], groups=1, padding=self.kernel_size // 2)[0]
        return img # [img, *args]
    
    @staticmethod
    def gaussian_kernel(size=5, sigma=1):
        size = int(size) // 2
        x, y, z = np.mgrid[-size:size+1, -size:size+1, -size:size+1]
        g = 1 / ((2.0 * np.pi * sigma ** 2) ** 1.5) * np.exp(-(x**2 + y**2 + z**2) / (2 * sigma ** 2))
        return torch.from_numpy(g / g.sum()).float()
    

class SimulateLowResolutionTorch:
    def __init__(self, scale, p=0.2):
        self.scale = scale
        self.p = p

    def __call__(self, img):
        if np.random.uniform() < self.p:
            scale_factor = np.random.uniform(*self.scale)
            size = img.size()[-3:]
            img = F.interpolate(img[None], scale_factor=scale_factor, mode='trilinear', align_corners=True)
            img = F.interpolate(img, size=size, mode='nearest')[0]
        return img # [img, *args]
    
class DownsampleForDS:
    def __init__(self, sizes, keys):
        self.sizes = sizes
        self.keys = keys
        self.ts = [T.Resize(s, mode='nearest') for s in sizes]

    def __call__(self, data):
        for k in self.keys:
            data[k] = [t(data[k]) for t in self.ts]
        return data

class Transform:
    def __init__(
        self, 
        intensityproperties=INTENSITYPROPERTIES, 
        target_resolution=(192,192,192),
        target_spacing=(1,1,1),
        patch_size=(96,96,96),
        num_patches=4,
        radius=4,
        x_key='Heart ROI',
        seg_key=None,
        y_key='HP'
    ):
        self.ct_normalization = CTNormalization(intensityproperties=intensityproperties)
        self.target_resolution = target_resolution
        self.target_spacing = target_spacing
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.radius = radius
        self.x_key = x_key
        self.seg_key = seg_key
        self.y_key = y_key

    def __call__(self, x, y, patches=False):
        heart_roi = x[self.x_key]
        heart_roi_norm = self.ct_normalization.run(torch.from_numpy(sitk.GetArrayFromImage(heart_roi)))
        heart_roi_norm =sitk.GetImageFromArray(heart_roi_norm.numpy())
        heart_roi_norm.CopyInformation(heart_roi)
        heart_roi_norm = resample_image(heart_roi_norm, out_spacing=self.target_spacing)
        heart_roi_norm = crop_or_pad_img(heart_roi_norm, self.target_resolution)
        
        # if self.seg_key is not None:
        #     heart_seg =  y[self.seg_key]
        #     heart_seg = resample_image(heart_seg, out_spacing=self.target_spacing, interpolator='NearestNeighbor')
        #     heart_seg = crop_or_pad_seg(heart_seg, self.target_resolution)
        
        data = {
            'heart_sitk': heart_roi_norm, 
            'heart_torch': sitk_to_torch(heart_roi_norm)[None],
            # 'seg_sitk': heart_seg if self.seg_key is not None else None, 
            # 'seg_torch': sitk_to_torch(heart_seg)[None] if self.seg_key is not None else None,
            # self.y_key: y, 
            # f'{self.y_key}_torch': torch.from_numpy(y_np)[None,None],
        }
        
        for yk in self.y_key:
            _y = y[yk]
            if 'hps' in yk.lower() or 'ms' in yk.lower():
            # if yk in ['HP', 'MS']:
                _y = _y.to_numpy().reshape(-1, 3)
                y_np = draw_spheres_from_physical_points(heart_roi_norm, _y, radius=self.radius, onehot=False)
                y_sitk = sitk.GetImageFromArray(y_np)
                y_sitk.CopyInformation(heart_roi_norm)
            elif 'ops' in yk.lower():
                data['pm'] = torch.tensor([[_y['5-377']]])
            else:
                y_res = resample_image(_y, out_spacing=self.target_spacing, interpolator='NearestNeighbor')
                y_res = crop_or_pad_seg(y_res, self.target_resolution)
                y_np = sitk.GetArrayFromImage(y_res)
            data[yk] = _y
            data[f'{yk}_torch'] = torch.from_numpy(y_np)[None,None]

            # if patches:
            #     extract_patch = lambda x, idx: x[idx[0,0]:idx[0,1],idx[1,0]:idx[1,1],idx[2,0]:idx[2,1]]
            #     max_patch_idx = [s-p for s, p in zip(self.target_resolution, self.patch_size)]
            #     heart_roi_patches, heart_seg_patches, y_patches = [], [], []
            #     for _ in range(self.num_patches):
            #         patch_idx = [np.random.randint(max_patch_idx[i]) for i in range(3)]
            #         patch_idx = np.array([[i, i+self.patch_size[j]] for j, i in enumerate(patch_idx)])
            #         heart_roi_patches.append(extract_patch(heart_roi_norm, patch_idx))
            #         if self.seg_key is not None:
            #             heart_seg_patches.append(extract_patch(heart_seg, patch_idx))
            #         y_patches.append(extract_patch(y_sitk, patch_idx))
            #     heart_roi_patches_torch = torch.stack([sitk_to_torch(t) for t in heart_roi_patches])
            #     if self.seg_key is not None:
            #         heart_seg_patches_torch = torch.stack([sitk_to_torch(t) for t in heart_seg_patches])
            #     y_patches_torch = torch.stack([sitk_to_torch(t) for t in y_patches])

            # data = {
            #     'heart_sitk': heart_roi_patches, 
            #     'heart_torch': heart_roi_patches_torch,
            #     'seg_sitk': heart_seg_patches if self.seg_key is not None else None, 
            #     'seg_torch': heart_seg_patches_torch if self.seg_key is not None else None,
            #     self.y_key: y, 
            #     f'{self.y_key}_torch':y_patches_torch,
            # }
        # else:
            # data = {
            #     'heart_sitk': heart_roi_norm, 
            #     'heart_torch': sitk_to_torch(heart_roi_norm)[None],
            #     'seg_sitk': heart_seg if self.seg_key is not None else None, 
            #     'seg_torch': sitk_to_torch(heart_seg)[None] if self.seg_key is not None else None,
            #     self.y_key: y, 
            #     f'{self.y_key}_torch': torch.from_numpy(y_np)[None,None],
            # }
        return data