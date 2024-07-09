import sys
sys.path.append('/mnt/ssd/git-repos/fedbiomed')
from fedbiomed.common.training_plans import TorchTrainingPlan
from fedbiomed.common.data import DataManager, MedicalFolderDataset
import os
os.environ['TOTALSEG_HOME_DIR'] = '/home/malte/.totalsegmentator'
os.environ['nnUNet_results'] = '/home/malte/.totalsegmentator/nnunet/results'
os.environ['nnUNet_raw'] = '/home/malte/.totalsegmentator/nnunet/results'
os.environ['nnUNet_preprocessed'] = '/home/malte/.totalsegmentator/nnunet/results'
from pathlib import Path
import shutil
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
import nibabel as nib
import SimpleITK as sitk

# from totalsegmentator.config import setup_nnunet, setup_totalseg
# from totalsegmentator.nnunet import nnUNet_predict_image
# from totalsegmentator.map_to_binary import class_map
# from totalsegmentator.alignment import as_closest_canonical
# from totalsegmentator.cropping import crop_to_mask
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

class FocusHeartTrainingPlan(TorchTrainingPlan):
    x_key = 'CT'
    y_keys = []

    class_map = {

        # classes of old TotalSegmentator v1
        "total_v1": {
            1: "spleen",
            2: "kidney_right",
            3: "kidney_left",
            4: "gallbladder",
            5: "liver",
            6: "stomach",
            7: "aorta",
            8: "inferior_vena_cava",
            9: "portal_vein_and_splenic_vein",
            10: "pancreas",
            11: "adrenal_gland_right",
            12: "adrenal_gland_left",
            13: "lung_upper_lobe_left",
            14: "lung_lower_lobe_left",
            15: "lung_upper_lobe_right",
            16: "lung_middle_lobe_right",
            17: "lung_lower_lobe_right",
            18: "vertebrae_L5",
            19: "vertebrae_L4",
            20: "vertebrae_L3",
            21: "vertebrae_L2",
            22: "vertebrae_L1",
            23: "vertebrae_T12",
            24: "vertebrae_T11",
            25: "vertebrae_T10",
            26: "vertebrae_T9",
            27: "vertebrae_T8",
            28: "vertebrae_T7",
            29: "vertebrae_T6",
            30: "vertebrae_T5",
            31: "vertebrae_T4",
            32: "vertebrae_T3",
            33: "vertebrae_T2",
            34: "vertebrae_T1",
            35: "vertebrae_C7",
            36: "vertebrae_C6",
            37: "vertebrae_C5",
            38: "vertebrae_C4",
            39: "vertebrae_C3",
            40: "vertebrae_C2",
            41: "vertebrae_C1",
            42: "esophagus",
            43: "trachea",
            44: "heart_myocardium",
            45: "heart_atrium_left",
            46: "heart_ventricle_left",
            47: "heart_atrium_right",
            48: "heart_ventricle_right",
            49: "pulmonary_artery",
            50: "brain",
            51: "iliac_artery_left",
            52: "iliac_artery_right",
            53: "iliac_vena_left",
            54: "iliac_vena_right",
            55: "small_bowel",
            56: "duodenum",
            57: "colon",
            58: "rib_left_1",
            59: "rib_left_2",
            60: "rib_left_3",
            61: "rib_left_4",
            62: "rib_left_5",
            63: "rib_left_6",
            64: "rib_left_7",
            65: "rib_left_8",
            66: "rib_left_9",
            67: "rib_left_10",
            68: "rib_left_11",
            69: "rib_left_12",
            70: "rib_right_1",
            71: "rib_right_2",
            72: "rib_right_3",
            73: "rib_right_4",
            74: "rib_right_5",
            75: "rib_right_6",
            76: "rib_right_7",
            77: "rib_right_8",
            78: "rib_right_9",
            79: "rib_right_10",
            80: "rib_right_11",
            81: "rib_right_12",
            82: "humerus_left",
            83: "humerus_right",
            84: "scapula_left",
            85: "scapula_right",
            86: "clavicula_left",
            87: "clavicula_right",
            88: "femur_left",
            89: "femur_right",
            90: "hip_left",
            91: "hip_right",
            92: "sacrum",
            93: "face",
            94: "gluteus_maximus_left",
            95: "gluteus_maximus_right",
            96: "gluteus_medius_left",
            97: "gluteus_medius_right",
            98: "gluteus_minimus_left",
            99: "gluteus_minimus_right",
            100: "autochthon_left",
            101: "autochthon_right",
            102: "iliopsoas_left",
            103: "iliopsoas_right",
            104: "urinary_bladder"
        },

        # classes of new TotalSegmentator v2
        "total": {
            1: "spleen",
            2: "kidney_right",
            3: "kidney_left",
            4: "gallbladder",
            5: "liver",
            6: "stomach",
            7: "pancreas",
            8: "adrenal_gland_right",
            9: "adrenal_gland_left",
            10: "lung_upper_lobe_left",
            11: "lung_lower_lobe_left",
            12: "lung_upper_lobe_right",
            13: "lung_middle_lobe_right",
            14: "lung_lower_lobe_right",
            15: "esophagus",
            16: "trachea",
            17: "thyroid_gland",
            18: "small_bowel",
            19: "duodenum",
            20: "colon",
            21: "urinary_bladder",
            22: "prostate",
            23: "kidney_cyst_left",
            24: "kidney_cyst_right",
            25: "sacrum",
            26: "vertebrae_S1",
            27: "vertebrae_L5",
            28: "vertebrae_L4",
            29: "vertebrae_L3",
            30: "vertebrae_L2",
            31: "vertebrae_L1",
            32: "vertebrae_T12",
            33: "vertebrae_T11",
            34: "vertebrae_T10",
            35: "vertebrae_T9",
            36: "vertebrae_T8",
            37: "vertebrae_T7",
            38: "vertebrae_T6",
            39: "vertebrae_T5",
            40: "vertebrae_T4",
            41: "vertebrae_T3",
            42: "vertebrae_T2",
            43: "vertebrae_T1",
            44: "vertebrae_C7",
            45: "vertebrae_C6",
            46: "vertebrae_C5",
            47: "vertebrae_C4",
            48: "vertebrae_C3",
            49: "vertebrae_C2",
            50: "vertebrae_C1",
            51: "heart",
            52: "aorta",
            53: "pulmonary_vein",
            54: "brachiocephalic_trunk",
            55: "subclavian_artery_right",
            56: "subclavian_artery_left",
            57: "common_carotid_artery_right",
            58: "common_carotid_artery_left",
            59: "brachiocephalic_vein_left",
            60: "brachiocephalic_vein_right",
            61: "atrial_appendage_left",
            62: "superior_vena_cava",
            63: "inferior_vena_cava",
            64: "portal_vein_and_splenic_vein",
            65: "iliac_artery_left",
            66: "iliac_artery_right",
            67: "iliac_vena_left",
            68: "iliac_vena_right",
            69: "humerus_left",
            70: "humerus_right",
            71: "scapula_left",
            72: "scapula_right",
            73: "clavicula_left",
            74: "clavicula_right",
            75: "femur_left",
            76: "femur_right",
            77: "hip_left",
            78: "hip_right",
            79: "spinal_cord",
            80: "gluteus_maximus_left",
            81: "gluteus_maximus_right",
            82: "gluteus_medius_left",
            83: "gluteus_medius_right",
            84: "gluteus_minimus_left",
            85: "gluteus_minimus_right",
            86: "autochthon_left",
            87: "autochthon_right",
            88: "iliopsoas_left",
            89: "iliopsoas_right",
            90: "brain",
            91: "skull",
            92: "rib_right_4",
            93: "rib_right_3",
            94: "rib_left_1",
            95: "rib_left_2",
            96: "rib_left_3",
            97: "rib_left_4",
            98: "rib_left_5",
            99: "rib_left_6",
            100: "rib_left_7",
            101: "rib_left_8",
            102: "rib_left_9",
            103: "rib_left_10",
            104: "rib_left_11",
            105: "rib_left_12",
            106: "rib_right_1",
            107: "rib_right_2",
            108: "rib_right_5",
            109: "rib_right_6",
            110: "rib_right_7",
            111: "rib_right_8",
            112: "rib_right_9",
            113: "rib_right_10",
            114: "rib_right_11",
            115: "rib_right_12",
            116: "sternum",
            117: "costal_cartilages"
        },
        # total_fast not extra class map, because easier to use just "total" for fast model
        "lung_vessels": {
            1: "lung_vessels",
            2: "lung_trachea_bronchia"
        },
        "covid": {
            1: "lung_covid_infiltrate",
        },
        "cerebral_bleed": {
            1: "intracerebral_hemorrhage",
        },
        "hip_implant": {
            1: "hip_implant",
        },
        "coronary_arteries": {
            1: "coronary_arteries",
        },
        "body": {
            1: "body_trunc",
            2: "body_extremities",
        },
        "pleural_pericard_effusion": {
            # 1: "lung_pleural",
            2: "pleural_effusion",
            3: "pericardial_effusion",
        },
        "liver_vessels": {
            1: "liver_vessels",
            2: "liver_tumor"
        },
        "vertebrae_body": {
            1: "vertebrae_body"
        },
        "heartchambers_highres": {
            1: "heart_myocardium", 
            2: "heart_atrium_left", 
            3: "heart_ventricle_left", 
            4: "heart_atrium_right", 
            5: "heart_ventricle_right", 
            6: "aorta", 
            7: "pulmonary_artery"
        },
        "appendicular_bones": {
            1: "patella",
            2: "tibia",
            3: "fibula",
            4: "tarsal",
            5: "metatarsal",
            6: "phalanges_feet",
            7: "ulna",
            8: "radius",
            9: "carpal",
            10: "metacarpal",
            11: "phalanges_hand"
        },
        # those classes need to be removed
        "appendicular_bones_auxiliary": {
            12: "humerus",
            13: "femur",
            14: "liver",
            15: "spleen"
        },
        "tissue_types": {
            1: "subcutaneous_fat",
            2: "torso_fat",
            3: "skeletal_muscle"
        },
        "face": {
            1: "face"
        },
        "test": {
            1: "carpal"
        }
    }


    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     setup_nnunet()
    #     setup_totalseg()
    
    def init_model(self, model_args):
        model = self.Net(model_args)
        return model

    def init_optimizer(self):
        optimizer = AdamW(self.model().parameters(), lr=1e-2)
        return optimizer

    # export TOTALSEG_HOME_DIR=/home/malte/.totalsegmentator
    # export nnUNet_results=/home/malte/.totalsegmentator/nnunet/results
    # export nnUNet_raw=/home/malte/.totalsegmentator/nnunet/results
    # export nnUNet_preprocessed=/home/malte/.totalsegmentator/nnunet/results
    def init_dependencies(self):
        deps = ["import torch",
                "import torch.nn as nn",
                "from torch.optim import AdamW",
                "from fedbiomed.common.data import MedicalFolderDataset",
                'import numpy as np',
                'import nibabel as nib',
                'import SimpleITK as sitk',
                'from pathlib import Path',
                'from scipy import ndimage',
                'import shutil',
                'import psutil',
                'from joblib import Parallel, delayed',
                # 'from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data'
                # 'from totalsegmentator.nnunet import nnUNet_predict_image',
                # 'from totalsegmentator.config import setup_nnunet, setup_totalseg',
                # 'from totalsegmentator.map_to_binary import class_map',
                # 'from totalsegmentator.alignment import as_closest_canonical',
                # 'from totalsegmentator.cropping import crop_to_mask'
                ]
        return deps

    class Net(nn.Module):
        def __init__(self, model_args: dict = {}):
            super().__init__()
            self.l = nn.Linear(1,1)

        def forward(self, x):
            return x
    
    @staticmethod
    def resample_img(img, zoom=0.5, order=0, nr_cpus=-1):
        """
        img: [x,y,z,(t)]
        zoom: 0.5 will halfen the image resolution (make image smaller)

        Resize numpy image array to new size.

        Faster than resample_img_nnunet.
        Resample_img_nnunet maybe slighlty better quality on CT (but not sure).
        
        Works for 2D and 3D and 4D images.
        """
        def _process_gradient(grad_idx):
            return ndimage.zoom(img[:, :, :, grad_idx], zoom, order=order)

        dim = len(img.shape)

        # Add dimesions to make each input 4D
        if dim == 2: 
            img = img[..., None, None]
        if dim == 3: 
            img = img[..., None]

        nr_cpus = psutil.cpu_count() if nr_cpus == -1 else nr_cpus
        img_sm = Parallel(n_jobs=nr_cpus)(delayed(_process_gradient)(grad_idx) for grad_idx in range(img.shape[3]))
        img_sm = np.array(img_sm).transpose(1, 2, 3, 0)  # grads channel was in front -> put to back
        # Remove added dimensions
        # img_sm = img_sm[:,:,:,0] if img_sm.shape[3] == 1 else img_sm  # remove channel dim if only 1 element
        if dim == 3:
            img_sm = img_sm[:,:,:,0]
        if dim == 2:
            img_sm = img_sm[:,:,0,0]
        return img_sm

    def change_spacing(self, img_in, new_spacing=1.25, target_shape=None, order=0, nr_cpus=1,
                    nnunet_resample=False, dtype=None, remove_negative=False, force_affine=None):
        """
        Resample nifti image to the new spacing (uses resample_img() internally).
        
        img_in: nifti image
        new_spacing: float or sequence of float
        target_shape: sequence of int (optional)
        order: resample order (optional)
        nnunet_resample: nnunet resampling will use order=0 sampling for z if very anisotropic. Sometimes results 
                        in a little bit less blurry results
        dtype: output datatype
        remove_negative: set all negative values to 0. Useful if resampling introduced negative values.
        force_affine: if you pass an affine then this will be used for the output image (useful if you have to make sure
                    that the resampled has identical affine to some other image. In this case also set target_shape.)

        Works for 2D and 3D and 4D images.

        If downsampling an image and then upsampling again to original resolution the resulting image can have
        a shape which is +-1 compared to original shape, because of rounding of the shape to int.
        To avoid this the exact output shape can be provided. Then new_spacing will be ignored and the exact
        spacing will be calculated which is needed to get to target_shape.
        In this case however the calculated spacing can be slighlty different from the desired new_spacing. This will
        result in a slightly different affine. To avoid this the desired affine can be writen by force with "force_affine".

        Note: Only works properly if affine is all 0 except for diagonal and offset (=no rotation and sheering)
        """
        data = img_in.get_fdata()  # quite slow
        old_shape = np.array(data.shape)
        img_spacing = np.array(img_in.header.get_zooms())

        if len(img_spacing) == 4:
            img_spacing = img_spacing[:3]  # for 4D images only use spacing of first 3 dims

        if type(new_spacing) is float:
            new_spacing = [new_spacing,] * 3   # for 3D and 4D
        new_spacing = np.array(new_spacing)

        if len(old_shape) == 2:
            img_spacing = np.array(list(img_spacing) + [new_spacing[2],])

        if target_shape is not None:
            # Find the right zoom to exactly reach the target_shape.
            # We also have to adapt the spacing to this new zoom.
            zoom = np.array(target_shape) / old_shape  
            new_spacing = img_spacing / zoom  
        else:
            zoom = img_spacing / new_spacing

        # copy very important; otherwise new_affine changes will also be in old affine
        new_affine = np.copy(img_in.affine)

        # This is only correct if all off-diagonal elements are 0
        # new_affine[0, 0] = new_spacing[0] if img_in.affine[0, 0] > 0 else -new_spacing[0]
        # new_affine[1, 1] = new_spacing[1] if img_in.affine[1, 1] > 0 else -new_spacing[1]
        # new_affine[2, 2] = new_spacing[2] if img_in.affine[2, 2] > 0 else -new_spacing[2]

        # This is the proper solution
        # Scale each column vector by the zoom of this dimension
        new_affine = np.copy(img_in.affine)
        new_affine[:3, 0] = new_affine[:3, 0] / zoom[0]
        new_affine[:3, 1] = new_affine[:3, 1] / zoom[1]
        new_affine[:3, 2] = new_affine[:3, 2] / zoom[2]

        # Just for information: How to get spacing from affine with rotation:
        # Calc length of each column vector:
        # vecs = affine[:3, :3]
        # spacing = tuple(np.sqrt(np.sum(vecs ** 2, axis=0)))

        # if nnunet_resample:
        #     new_data, _ = resample_img_nnunet(data, None, img_spacing, new_spacing)
        # else:
        #     # if cupy_available and cucim_available:
        #     #     new_data = resample_img_cucim(data, zoom=zoom, order=order, nr_cpus=nr_cpus)  # gpu resampling
        #     # else:
        new_data = self.resample_img(data, zoom=zoom, order=order, nr_cpus=nr_cpus)  # cpu resampling
            
        if remove_negative:
            new_data[new_data < 1e-4] = 0

        if dtype is not None:
            new_data = new_data.astype(dtype)

        if force_affine is not None:
            new_affine = force_affine

        return nib.Nifti1Image(new_data, new_affine)
    
    @staticmethod
    def get_bbox_from_mask(mask, outside_value=-900, addon=0):
        if type(addon) is int:
            addon = [addon] * 3
        if (mask > outside_value).sum() == 0: 
            print("WARNING: Could not crop because no foreground detected")
            minzidx, maxzidx = 0, mask.shape[0]
            minxidx, maxxidx = 0, mask.shape[1]
            minyidx, maxyidx = 0, mask.shape[2]
        else:
            mask_voxel_coords = np.where(mask > outside_value)
            minzidx = int(np.min(mask_voxel_coords[0])) - addon[0]
            maxzidx = int(np.max(mask_voxel_coords[0])) + 1 + addon[0]
            minxidx = int(np.min(mask_voxel_coords[1])) - addon[1]
            maxxidx = int(np.max(mask_voxel_coords[1])) + 1 + addon[1]
            minyidx = int(np.min(mask_voxel_coords[2])) - addon[2]
            maxyidx = int(np.max(mask_voxel_coords[2])) + 1 + addon[2]

        # Avoid bbox to get out of image size
        s = mask.shape
        minzidx = max(0, minzidx)
        maxzidx = min(s[0], maxzidx)
        minxidx = max(0, minxidx)
        maxxidx = min(s[1], maxxidx)
        minyidx = max(0, minyidx)
        maxyidx = min(s[2], maxyidx)

        return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

    @staticmethod
    def crop_to_bbox(image, bbox):
        """
        image: 3d nd.array
        bbox: list of lists [[minx_idx, maxx_idx], [miny_idx, maxy_idx], [minz_idx, maxz_idx]]
            Indices of bbox must be in voxel coordinates  (not in world space)
        """
        assert len(image.shape) == 3, "only supports 3d images"
        return image[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]

    def crop_to_bbox_nifti(self, image: nib.Nifti1Image, bbox, dtype=None) -> nib.Nifti1Image:
        """
        Crop nifti image to bounding box and adapt affine accordingly

        image: nib.Nifti1Image
        bbox: list of lists [[minx_idx, maxx_idx], [miny_idx, maxy_idx], [minz_idx, maxz_idx]]
            Indices of bbox must be in voxel coordinates  (not in world space)
        dtype: dtype of the output image

        returns: nib.Nifti1Image
        """
        assert len(image.shape) == 3, "only supports 3d images"
        data = image.get_fdata()

        # Crop the image
        data_cropped = self.crop_to_bbox(data, bbox)

        # Update the affine matrix
        affine = np.copy(image.affine)
        affine[:3, 3] = np.dot(affine, np.array([bbox[0][0], bbox[1][0], bbox[2][0], 1]))[:3]

        data_type = image.dataobj.dtype if dtype is None else dtype
        return nib.Nifti1Image(data_cropped.astype(data_type), affine)

    def crop_to_mask(self, img_in, mask_img, addon=[0,0,0], dtype=None, verbose=False):
        """
        Crops a nifti image to a mask and adapts the affine accordingly.

        img_in: nifti image 
        mask_img: nifti image 
        addon = addon in mm along each axis
        dtype: output dtype

        Returns a nifti image.
        """
        # This is needed for body mask with sometimes does not have the same shape as the 
        # input image because it was generated on a lower resolution.
        # (normally the body mask should be resampled to the original resolution, but it 
        # might have been generated by a different program)
        # This is quite slow for large images. Since normally not needed we remove it.
        # 
        # print("Transforming crop mask to img space:")  
        # print(f"  before: {mask_img.shape}")
        # mask_img = nibabel.processing.resample_from_to(mask_img, img_in, order=0)
        # print(f"  after: {mask_img.shape}")

        mask = mask_img.get_fdata()
        
        addon = (np.array(addon) / img_in.header.get_zooms()).astype(int)  # mm to voxels
        bbox = self.get_bbox_from_mask(mask, outside_value=0, addon=addon)

        img_out = self.crop_to_bbox_nifti(img_in, bbox, dtype)
        return img_out, bbox

    @staticmethod
    def as_closest_canonical(img_in):
        """
        Convert the given nifti file to the closest canonical nifti file.
        """
        return nib.as_closest_canonical(img_in)
    
    @staticmethod
    def undo_canonical(img_can, img_orig):
        """
        Inverts nib.to_closest_canonical()

        img_can: the image we want to move back
        img_orig: the original image because transforming to canonical

        returns image in original space

        https://github.com/nipy/nibabel/issues/1063
        """
        from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform

        img_ornt = io_orientation(img_orig.affine)
        ras_ornt = axcodes2ornt("RAS")

        to_canonical = img_ornt  # Same as ornt_transform(img_ornt, ras_ornt)
        from_canonical = ornt_transform(ras_ornt, img_ornt)

        # Same as as_closest_canonical
        # img_canonical = img_orig.as_reoriented(to_canonical)

        return img_can.as_reoriented(from_canonical)

    @staticmethod
    def check_if_shape_and_affine_identical(img_1, img_2):
        
        if not np.array_equal(img_1.affine, img_2.affine):
            print("Affine in:")
            print(img_1.affine)
            print("Affine out:")
            print(img_2.affine)
            print("Diff:")
            print(np.abs(img_1.affine-img_2.affine))
            print("WARNING: Output affine not equal to input affine. This should not happen.")

        if img_1.shape != img_2.shape:
            print("Shape in:")
            print(img_1.shape)
            print("Shape out:")
            print(img_2.shape)
            print("WARNING: Output shape not equal to input shape. This should not happen.")

    @staticmethod
    def download_ts():
        if not Path('.totalsegmentator').exists():
            import requests
            from zipfile import ZipFile
            from io import BytesIO
            link = 'https://heibox.uni-heidelberg.de/seafhttp/files/c48f165b-8e75-4236-b572-b6ae7cd14cd3/totalsegmentator.zip'
            r = requests.get(link)
            z = ZipFile(BytesIO(r.content))
            z.extractall(".")

    def training_data(self, batch_size=1):
        dataset = MedicalFolderDataset(
            root=self.dataset_path,
            data_modalities=self.x_key,
            target_modalities=self.y_keys,
            transform=None,
            target_transform=None,
            demographics_transform=None
        )
        loader_arguments = {'batch_size': batch_size, 'shuffle': True}
        return DataManager(dataset, **loader_arguments)

    def training_step(self, data, target):
        return torch.tensor([0])

    def testing_step(self, data, target):
        self.download_ts()
        import os
        os.environ['nnUNet_results'] = '.totalsegmentator/nnunet/results'
        os.environ['nnUNet_raw'] = '.totalsegmentator/nnunet/results'
        os.environ['nnUNet_preprocessed'] = '.totalsegmentator/nnunet/results'
        # import pdb;pdb.set_trace()
        from nnunetv2.utilities.file_path_utilities import get_output_folder
        from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
        inputs = data[0][self.x_key]
        fname = Path(inputs.meta['filename_or_obj'])

        file_in = Path(fname)
        img_in_orig = nib.load(file_in)
        img_in = nib.Nifti1Image(img_in_orig.get_fdata(), img_in_orig.affine)

        img_in = nib.as_closest_canonical(img_in)

        img_in_shape = img_in.shape
        img_in_zooms = img_in.header.get_zooms()
        resample = 6.0
        nr_threads_resampling = 1
        img_in_rsp = self.change_spacing(img_in, [resample, resample, resample], order=3, dtype=np.int32, nr_cpus=nr_threads_resampling)

        model = "3d_fullres"
        trainer = "nnUNetTrainer_4000epochs_NoMirroring"
        task_id = 298
        plans = "nnUNetPlans"

        model_folder = get_output_folder(task_id, trainer, plans, model)

        step_size = 0.5
        disable_tta = True # not tta
        verbose = False
        save_probabilities = False
        continue_prediction = False
        chk = "checkpoint_final.pth"
        npp = 3 # num_threads_preprocessing
        nps = 2 # num_threads_nifti_save
        prev_stage_predictions = None
        num_parts = 1
        part_id = 0
        allow_tqdm = False # not quiet
        folds = None

        device = torch.device('cuda')

        tmp_dir = Path('./tmp12345342f4f')
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir()
        dir_in = str(tmp_dir)
        dir_out = str(tmp_dir)

        nib.save(img_in_rsp, tmp_dir / "s01_0000.nii.gz")
        nr_voxels_thr = 256*256*900
        img_parts = ["s01"]
        ss = img_in_rsp.shape

        predict_from_raw_data(dir_in,
                                dir_out,
                                model_folder,
                                folds,
                                step_size,
                                use_gaussian=True,
                                use_mirroring=not disable_tta,
                                perform_everything_on_gpu=True,
                                verbose=verbose,
                                save_probabilities=save_probabilities,
                                overwrite=not continue_prediction,
                                checkpoint_name=chk,
                                num_processes_preprocessing=npp,
                                num_processes_segmentation_export=nps,
                                folder_with_segs_from_prev_stage=prev_stage_predictions,
                                num_parts=num_parts,
                                part_id=part_id,
                                device=device)

        # predictor = nnUNetPredictor(
        #     tile_step_size=step_size,
        #     use_gaussian=True,
        #     use_mirroring=not disable_tta,
        #     perform_everything_on_gpu=True,
        #     device=device,
        #     verbose=verbose,
        #     verbose_preprocessing=verbose,
        #     allow_tqdm=allow_tqdm
        # )
        # predictor.initialize_from_trained_model_folder(
        #     model_folder,
        #     use_folds=folds,
        #     checkpoint_name=chk,
        # )

        
        # predictor.predict_from_files(dir_in, dir_out,
        #                             save_probabilities=save_probabilities, overwrite=not continue_prediction,
        #                             num_processes_preprocessing=npp, num_processes_segmentation_export=nps,
        #                             folder_with_segs_from_prev_stage=prev_stage_predictions, 
        #                             num_parts=num_parts, part_id=part_id)
        
        img_pred = nib.load(tmp_dir / "s01.nii.gz")

        # Currently only relevant for T304 (appendicular bones)
        # img_pred = remove_auxiliary_labels(img_pred, task_name)

        resample = 6.0
        img_pred = self.change_spacing(img_pred, [resample, resample, resample], img_in_shape,
                                        order=0, dtype=np.uint8, nr_cpus=nr_threads_resampling, 
                                        force_affine=img_in.affine)

        img_pred = self.undo_canonical(img_pred, img_in_orig)

        self.check_if_shape_and_affine_identical(img_in_orig, img_pred)

        img_data = img_pred.get_fdata().astype(np.uint8)

        organ_seg = nib.Nifti1Image(img_data, img_pred.affine)

        # print(fname)
        # quiet = False
        # verbose = True
        # output_type = "nifti"
        # device = "cuda"
        # nr_thr_resamp = 1
        # crop_addon = [5, 5, 5]
        
        # organ_seg = nnUNet_predict_image(fname, None, 298, model="3d_fullres", folds=[0],
        #                                 trainer="nnUNetTrainer_4000epochs_NoMirroring", tta=False, multilabel_image=True, resample=6.0,
        #                                 crop=None, crop_path=None, task_name="total", nora_tag="None", preview=False, 
        #                                 save_binary=False, nr_threads_resampling=nr_thr_resamp, nr_threads_saving=1, 
        #                                 crop_addon=crop_addon, output_type=output_type, statistics=False,
        #                                 quiet=quiet, verbose=verbose, test=0)
        class_map_inv = {v: k for k, v in self.class_map["total"].items()}
        crop_mask = np.zeros(organ_seg.shape, dtype=np.uint8)
        organ_seg_data = organ_seg.get_fdata()
        roi = 'heart'
        crop_mask[organ_seg_data == class_map_inv[roi]] = 1
        crop_mask = nib.Nifti1Image(crop_mask, organ_seg.affine)
        crop_addon = [20,20,20]

        fname = Path(fname)
        img_in_orig = nib.load(fname)
        img_in = nib.Nifti1Image(img_in_orig.get_fdata(), img_in_orig.affine)
        img_in, bbox = self.crop_to_mask(img_in, crop_mask, addon=crop_addon, dtype=np.int32, verbose=verbose)
        img_in = self.as_closest_canonical(img_in)

        img_sitk = sitk.ReadImage(fname)
        img_sitk = img_sitk[bbox[0][0]:bbox[0][1],bbox[1][0]:bbox[1][1],bbox[2][0]:bbox[2][1]]
        
        pid_path = Path(fname).parents[1]
        (pid_path / 'Heart ROI').mkdir(exist_ok=True)
        sitk.WriteImage(img_sitk, pid_path / 'Heart ROI' / 'img.nii.gz')
        torch.save(bbox, pid_path / 'Heart ROI' / 'bbox.pt')
        shutil.rmtree(tmp_dir)
        return {'Loss': 0.}

class FocusSegTrainingPlan(TorchTrainingPlan):
    
    def init_model(self, model_args):
        model = self.Net(model_args)
        return model

    def init_optimizer(self):
        optimizer = AdamW(self.model().parameters(), lr=1e-2)
        return optimizer

    def init_dependencies(self):
        deps = ["import torch",
                "import torch.nn as nn",
                "from torch.optim import AdamW",
                "from fedbiomed.common.data import MedicalFolderDataset",
                'import SimpleITK as sitk',
                'from pathlib import Path']
        return deps

    class Net(nn.Module):
        def __init__(self, model_args: dict = {}):
            super().__init__()
            self.l = nn.Linear(1,1)

        def forward(self, x):
            return x

    def training_data(self, batch_size=1):
        dataset = MedicalFolderDataset(
            root=self.dataset_path,
            data_modalities=['Heart ROI'],
            target_modalities=['Calcification'],
            transform=None,
            target_transform=None,
            demographics_transform=None
        )
        loader_arguments = {'batch_size': batch_size, 'shuffle': True}
        return DataManager(dataset, **loader_arguments)

    def training_step(self, data, target):
        return torch.tensor([0])

    def testing_step(self, data, target):
        inputs = data[0]['Heart ROI']
        fname = Path(inputs.meta['filename_or_obj'])
        bbox = torch.load(fname.parents[0] / 'bbox.pt')
        y = sitk.ReadImage(target['Calcification'].meta['filename_or_obj'])
        y_foc = y[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
        pid_path = fname.parents[1]
        (pid_path / 'Calcification focused').mkdir(exist_ok=True)
        sitk.WriteImage(y_foc, pid_path / 'Calcification focused' / 'seg.nii.gz')
        return {'Loss': 0.}
    
class SegmentHeartTrainingPlan(TorchTrainingPlan):
    x_key = 'Heart ROI'
    y_keys = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        task_id = 301
        trainer = "nnUNetTrainer"
        model = "3d_fullres"
        plans = "nnUNetPlans"

        self.download_ts()
        import os
        os.environ['nnUNet_results'] = '/fedbiomed/.totalsegmentator/nnunet/results'
        os.environ['nnUNet_raw'] = '/fedbiomed/.totalsegmentator/nnunet/results'
        os.environ['nnUNet_preprocessed'] = '/fedbiomed/.totalsegmentator/nnunet/results'

        self.config_dir = get_output_folder(task_id, trainer, plans, model)

        _, self.preprocess_fn, _ = self.configure_nnunet(self.config_dir, nnUNetTrainer)
    
    def init_model(self, model_args):
        network, _, _ = self.configure_nnunet(self.config_dir, nnUNetTrainer)
        return network

    def init_optimizer(self):
        optimizer = AdamW(self.model().parameters(), lr=1e-2)
        return optimizer

    @staticmethod
    def download_ts():
        if not Path('.totalsegmentator').exists():
            import requests
            from zipfile import ZipFile
            from io import BytesIO
            link = 'https://heibox.uni-heidelberg.de/seafhttp/files/f0444e2d-7906-41b2-8eef-4078713c91eb/totalsegmentator.zip'
            r = requests.get(link)
            z = ZipFile(BytesIO(r.content))
            z.extractall(".")

    # export TOTALSEG_HOME_DIR=/home/malte/.totalsegmentator
    # export nnUNet_results=/home/malte/.totalsegmentator/nnunet/results
    # export nnUNet_raw=/home/malte/.totalsegmentator/nnunet/results
    # export nnUNet_preprocessed=/home/malte/.totalsegmentator/nnunet/results
    def init_dependencies(self):
        deps = ["import torch",
                "import torch.nn as nn",
                "from torch.optim import AdamW",
                "from fedbiomed.common.data import MedicalFolderDataset",
                'import numpy as np',
                'import nibabel as nib',
                'import SimpleITK as sitk',
                'import json',
                'from pathlib import Path',
                'from acvl_utils.cropping_and_padding.padding import pad_nd_image',
                'from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer',
                'from nnunetv2.utilities.plans_handling.plans_handler import PlansManager',
                'from nnunetv2.utilities.file_path_utilities import get_output_folder',
                'from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor',
                'from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero',
                'from nnunetv2.inference.sliding_window_prediction import compute_gaussian',
                'from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window',
                'from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape']
        return deps

    @staticmethod
    def resample_image(
            itk_image,
            new_size=None,
            out_spacing=None,
            interpolator="BSpline"  # "BSpline", "NearestNeighbor" is_label=False
    ) -> sitk.Image:
        # https://www.programcreek.com/python/example/96383/SimpleITK.sitkNearestNeighbor

        if out_spacing is None:
            out_spacing = [
                sz * spc / nsz for nsz, sz, spc in
                zip(new_size, itk_image.GetSize(), itk_image.GetSpacing())
            ]

        original_spacing = itk_image.GetSpacing()
        original_size = itk_image.GetSize()
        
        out_size = [
            int(np.round(original_size[i] * (original_spacing[i] / out_spacing[i]))) for i in range(len(original_size))
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(out_spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(itk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

        interpolator = getattr(sitk, f"sitk{interpolator}")
        resample.SetInterpolator(interpolator)

        return resample.Execute(itk_image)

    def preprocess_case(self, img, plans_manager, configuration_manager):
        preprocessor = DefaultPreprocessor()
        properties = {}
        img_np = sitk.GetArrayFromImage(img)[None].astype(np.float32)
        original_spacing = img.GetSpacing()[::-1]
        properties['spacing'] = original_spacing
        shape_before_cropping = img_np.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping

        seg = None
        img_np, seg, bbox = crop_to_nonzero(img_np, seg)
        properties['bbox_used_for_cropping'] = bbox
        # TODO: bbox here is z,y,x (better x,y,z?) because of orientation in nnUNet
        img = img[bbox[2][0]:bbox[2][1],bbox[1][0]:bbox[1][1],bbox[0][0]:bbox[0][1]]
        properties['spacing_after_cropping_and_before_resampling'] = img.GetSpacing()[::-1]
        properties['shape_after_cropping_and_before_resampling'] = img_np.shape[1:]
        properties['patch_size'] = configuration_manager.patch_size

        target_spacing = configuration_manager.spacing
        new_shape = compute_new_shape(img_np.shape[1:], original_spacing, target_spacing)
        # here seg must be none!
        img_np = preprocessor._normalize(img_np, np.zeros_like(img_np), configuration_manager,
                                plans_manager.foreground_intensity_properties_per_channel)
        img_np = configuration_manager.resampling_fn_data(img_np, new_shape, original_spacing, target_spacing)

        img_res = self.resample_image(img, new_size=new_shape[::-1])

        x = torch.from_numpy(img_np)
        x, slicer_revert_padding = pad_nd_image(
            image=x, 
            new_shape=configuration_manager.patch_size,
            mode='constant', 
            kwargs={'value': 0}, 
            return_slicer=True,
            shape_must_be_divisible_by=None
        )
        properties['slicer_revert_padding'] = slicer_revert_padding
        return x.float(), img, img_res, properties

    @staticmethod
    def postprocess_case(logits, properties, configuration_manager):
        slicer_revert_padding = properties['slicer_revert_padding']
        pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        # TODO: the syntax was not accepted: *slicer_revert_padding
        pred = pred[slicer_revert_padding]
        pred_np = pred.cpu().detach().numpy()
        new_shape = properties['shape_after_cropping_and_before_resampling']
        original_spacing = configuration_manager.spacing
        target_spacing = properties['spacing_after_cropping_and_before_resampling']
        mask = configuration_manager.resampling_fn_seg(pred_np, new_shape, original_spacing, target_spacing)
        return mask

    def configure_nnunet(self, config_dir, trainer_class):
        dataset_json = f'{config_dir}/dataset.json'
        with open(dataset_json, 'r') as f:
            dataset_json = json.load(f)
        plans_file = f'{config_dir}/plans.json'
        plans_manager = PlansManager(plans_file)
        configuration_manager = plans_manager.get_configuration('3d_fullres')
        num_input_channels = 1
        network = trainer_class.build_network_architecture(plans_manager, dataset_json, configuration_manager,
                                                           num_input_channels, enable_deep_supervision=False)
        ckpt = torch.load(f'{config_dir}/fold_0/checkpoint_final.pth')
        network.load_state_dict(ckpt['network_weights'])
        preprocess_fn = lambda img_np: self.preprocess_case(img_np, plans_manager, configuration_manager)
        postprocess_fn = lambda l, p: self.postprocess_case(l, p, configuration_manager)
        return network, preprocess_fn, postprocess_fn

    @staticmethod
    @torch.no_grad()
    def predict_logits_sliding_window(network, data, num_segmentation_heads, patch_size, slicer_revert_padding, device=torch.device('cuda')):
        image_size = data.shape[1:]
        tile_step_size = 0.5
        steps = compute_steps_for_sliding_window(image_size, patch_size, tile_step_size)
        slicers = []
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicers.append(
                        tuple([slice(None), *[slice(si, si + ti) for si, ti in
                            zip((sx, sy, sz), patch_size)]]))

        use_gaussian = True
        predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]),
                                                    dtype=torch.half,
                                                    device=device)
        n_predictions = torch.zeros(data.shape[1:], dtype=torch.half,
                                    device=device)
        if use_gaussian:
            gaussian = compute_gaussian(tuple(patch_size), sigma_scale=1. / 8,
                                        value_scaling_factor=1000,
                                        device=device)
        
        for sl in slicers:
            workon = data[sl][None]
            workon = workon.to(device, non_blocking=False)

            with torch.no_grad():
                prediction = network(workon)[0]

            predicted_logits[sl] += (prediction * gaussian if use_gaussian else prediction)
            n_predictions[sl[1:]] += (gaussian if use_gaussian else 1)

        predicted_logits /= n_predictions
        # empty_cache(self.device)
        return predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]

    def training_data(self, batch_size=1):
        dataset = MedicalFolderDataset(
            root=self.dataset_path,
            data_modalities=self.x_key,
            target_modalities=self.y_keys,
            transform=None,
            target_transform=None,
            demographics_transform=None
        )
        loader_arguments = {'batch_size': batch_size, 'shuffle': True}
        return DataManager(dataset, **loader_arguments)

    def training_step(self, data, target):
        return torch.tensor([0])

    def testing_step(self, data, target):
        
        inputs = data[0][self.x_key]
        fname = Path(inputs.meta['filename_or_obj'])
        print(fname)
        img_sitk = sitk.ReadImage(fname)
        network = self.model()
        network.cuda()
        x, x_sitk, x_sitk_res, properties = self.preprocess_fn(img_sitk)
        heart_logits = self.predict_logits_sliding_window(network, x, 8, properties['patch_size'], properties['slicer_revert_padding']) # , device=self._device)
        heart_pred = torch.argmax(torch.softmax(heart_logits, dim=0), dim=0)
        heart_pred = heart_pred.cpu().numpy().astype(np.uint8)
        heart_pred_sitk = sitk.GetImageFromArray(heart_pred)
        heart_pred_sitk.CopyInformation(x_sitk_res)
        
        pid_path = Path(fname).parents[1]
        (pid_path / 'Heart Seg').mkdir(exist_ok=True)
        sitk.WriteImage(heart_pred_sitk, pid_path / 'Heart Seg' / 'img.nii.gz')
        return {'Loss': 0.}

if __name__ == '__main__':
    import os
    os.environ['MQTT_USERNAME'] = 'malte.toelle.heidelberg'
    os.environ['MQTT_PASSWORD'] = 'nvuisdf8rhqw'
    os.environ['MQTT_CERT_PATH'] = '/mnt/ssd/git-repos/dl-data-preprocessing/floto/mqtt/combined_chain.pem'

    from argparse import ArgumentParser
    from federated import federated_experiment
    parser = ArgumentParser()
    parser.add_argument('--training_plan')
    parser.add_argument('--config', '-c')
    parser.add_argument('--locations', '-l', nargs='+', default=None)
    parser.add_argument('--mode', '-m', default='test')
    parser.add_argument('--tags', '-t', nargs='+')
    parser.add_argument('--exp_name', '-e', default='preprop')
    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('--ckpt')
    args = parser.parse_args()

    if args.training_plan == 'focus':
        training_plan = FocusHeartTrainingPlan
    elif args.training_plan == 'seg':
        training_plan = SegmentHeartTrainingPlan
    elif args.training_plan == 'focus_seg':
        training_plan = FocusSegTrainingPlan
    else:
        raise ValueError
    
    federated_experiment(training_plan=training_plan, args=args)