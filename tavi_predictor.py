import os
from pathlib import Path
import numpy as np
import torch
import SimpleITK as sitk
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.inferers import sliding_window_inference
import cc3d
from scipy import ndimage
# from utils import (
#     mps_from_pts,
#     find_largest_volume,
#     find_center_of_mass
# )
from nnunet import nnunet_configuration
from transforms import (
    resample_image,
    CTNormalization,
    crop_or_pad_img,
    crop_or_pad_seg,
)
from predict_heart_logits import (
    crop_heart,
    heart_logits_from_cropped
)

def find_center_of_mass(seg: np.ndarray) -> np.array:
    #seg_labeled = ndimage.label(seg)[0]
    centers = ndimage.center_of_mass(seg, seg, np.unique(seg)[1:])
    return np.array(centers)

def find_largest_volume(seg: np.ndarray) -> np.ndarray:
    seg_c = np.zeros_like(seg)
    for c in np.unique(seg)[1:]:
        labels = cc3d.connected_components(seg==c)
        try:
            # leave out 0 because that is usually largest component
            idx = np.bincount(labels.flatten())[1:].argmax()
            seg_c[labels == idx+1] = c
        except: # Throws error if there is only one component
            pass
    return seg_c.astype(int)

def mps_from_pts(pts, path):
    xml_from_pt = lambda i, pt: f"""<point>
                <id>{i}</id>
                <specification>0</specification>
                <x>{pt[0]}</x>
                <y>{pt[1]}</y>
                <z>{pt[2]}</z>
            </point>
            """
    mps_string = f"""<?xml version="1.0" encoding="UTF-8"?>
<point_set_file>
    <file_version>0.1</file_version>
    <point_set>
        <time_series>
            <time_series_id>0</time_series_id>
            <Geometry3D ImageGeometry="false" FrameOfReferenceID="0">
                <IndexToWorld type="Matrix3x3" m_0_0="1" m_0_1="0" m_0_2="0" m_1_0="0" m_1_1="1" m_1_2="0" m_2_0="0" m_2_1="0" m_2_2="1"/>
                <Offset type="Vector3D" x="0" y="0" z="0"/>
                <Bounds>
                    <Min type="Vector3D" x="{np.min(pts[:,0])}" y="{np.min(pts[:,1])}" z="{np.min(pts[:,2])}"/>
                    <Max type="Vector3D" x="{np.max(pts[:,0])}" y="{np.max(pts[:,1])}" z="{np.max(pts[:,2])}"/>
                </Bounds>
            </Geometry3D>
            {''.join([xml_from_pt(i, pt) for i, pt in enumerate(pts)])}
        </time_series>
    </point_set>
</point_set_file>"""
    with open(path, 'w') as f:
        f.write(mps_string)

class TAVIPredictor:
    intensityproperties = {
        "max": 2176.0,
        "mean": 373.7686767578125,
        "median": 352.0,
        "min": -813.0,
        "percentile_00_5": -42.0,
        "percentile_99_5": 878.0,
        "std": 166.8054656982422
    }

    def __init__(
            self, 
            fname, 
            model_type='nnunet',
            hps_ckpt_fname='./checkpoints/federated_hps_conditioned_on_heart.pt', #'experiments/federated/hps-conditioned-on-heart/1696862910.235817/breakpoint_0005/aggregated_params_current.pt', #'checkpoints/tavi_predictor/hps_final_ckpt.pt',
            ms_ckpt_fname='./checkpoints/federated_ms_conditioned_on_heart.pt', #'experiments/federated/ms-conditioned-on-heart/1696590758.614868/breakpoint_0009/aggregated_params_current.pt',#'checkpoints/tavi_predictor/ms_final_ckpt.pt',
            calc_ckpt_fname='./checkpoints/federated_calc_conditioned_on_heart.pt', # 'experiments/federated/calc-conditioned-on-heart/1696608010.868197/breakpoint_0009/aggregated_params_current.pt',#'checkpoints/tavi_predictor/calc_final_ckpt.pt',
            tmp_dir='./tmp',
            device='cuda'
        ):
        self.fname = Path(fname)
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(exist_ok=True)
        self.heart_roi_fname = None
        self.heart_seg_fname = None
        self.hps_fname = None
        self.ms_fname = None
        self.calc_fname = None

        self.model_type = model_type
        self.hps_ckpt_fname = hps_ckpt_fname
        self.ms_ckpt_fname = ms_ckpt_fname
        self.calc_ckpt_fname = calc_ckpt_fname

        self.device = device
        pass

    def set_fname(self, fname):
        self.fname = Path(fname)
        self.heart_roi_fname = None
        self.heart_seg_fname = None
        self.hps_fname = None
        self.ms_fname = None
        self.calc_fname = None

    @torch.no_grad()
    def focus_heart(self, fname='heart_roi.nii.gz'):
        heart_roi_sitk, bbox = crop_heart(self.fname)
        (self.tmp_dir / 'Heart ROI').mkdir(exist_ok=True)
        self.heart_roi_fname = self.tmp_dir / 'Heart ROI' / fname
        sitk.WriteImage(heart_roi_sitk, self.heart_roi_fname)
        torch.save(bbox, self.tmp_dir / 'Heart ROI' / 'bbox.pt')

    @torch.no_grad()
    def segment_heart(self, fname='heart_seg.nii.gz'):
        if self.heart_roi_fname is None:
            self.focus_heart()
        heart_roi_sitk = sitk.ReadImage(self.heart_roi_fname)
        heart_logits, _, heart_roi_sitk_res = heart_logits_from_cropped(heart_roi_sitk)
        heart_pred = torch.argmax(torch.softmax(heart_logits, dim=0), dim=0)
        heart_pred = heart_pred.cpu().numpy().astype(np.uint8)
        heart_pred_sitk = sitk.GetImageFromArray(heart_pred)
        heart_pred_sitk.CopyInformation(heart_roi_sitk_res)
        (self.tmp_dir / 'Heart Seg').mkdir(exist_ok=True)
        self.heart_seg_fname = self.tmp_dir / 'Heart Seg' / fname
        sitk.WriteImage(heart_pred_sitk, self.heart_seg_fname)

    @staticmethod
    def get_ckpt(path, task):
        ckpt = torch.load(path)
        ckpt = {k.replace('unet.', '').replace('model.','').replace('swin_unetr.',''): v for k, v in ckpt.items()}
        ckpt = {k.replace('model.', ''): v for k, v in ckpt.items()}
        for i in range(5):
            k1 = f'seg_layers.{task}.{i}.weight'
            k2 = f'seg_layers.{task}.{i}.bias'
            if k1 in ckpt:
                ckpt[f'decoder.seg_layers.{i}.weight'] = ckpt[k1]
                ckpt[f'decoder.seg_layers.{i}.bias'] = ckpt[k2]
        if f'outs.{task}.conv.conv.weight' in ckpt:
            ckpt['out.conv.conv.weight'] = ckpt[f'outs.{task}.conv.conv.weight']
            ckpt['out.conv.conv.bias'] = ckpt[f'outs.{task}.conv.conv.bias']
        ckpt = {k: v for k, v in ckpt.items() if not k.startswith('seg_layers') and not k.startswith('outs')}
        for k in ['fed_task', 'patches', 'deep_supervision', 'condition_on_seg', 'output_seg']:
            if k in ckpt:
                del ckpt[k]
        return ckpt

    @torch.no_grad()
    def hinge_points(self):
        if self.model_type == 'nnunet':
            model = nnunet_configuration(num_segmentation_heads=6, num_input_channels=2).to(self.device)
        elif self.model_type == 'swin_unetr':
            model = SwinUNETR(
                img_size=(96,96,96),
                in_channels=2,
                out_channels=6,
                feature_size=48,
                use_checkpoint=False,
            ).to(self.device)
        model.eval()
        # ckpt = torch.load(self.hps_ckpt_fname)
        # ckpt = {k.replace('unet.', '').replace('model.','').replace('swin_unetr.',''): v for k, v in ckpt.items()}
        # ckpt = {k.replace('model.', ''): v for k, v in ckpt.items()}
        # for i in range(5):
        #     k1 = f'seg_layers.hps.{i}.weight'
        #     k2 = f'seg_layers.hps.{i}.bias'
        #     if k1 in ckpt:
        #         ckpt[f'decoder.seg_layers.{i}.weight'] = ckpt[k1]
        #         ckpt[f'decoder.seg_layers.{i}.bias'] = ckpt[k2]
        # if 'outs.hps.conv.conv.weight' in ckpt:
        #     ckpt['out.conv.conv.weight'] = ckpt['outs.hps.conv.conv.weight']
        #     ckpt['out.conv.conv.bias'] = ckpt['outs.hps.conv.conv.bias']
        # ckpt = {k: v for k, v in ckpt.items() if not k.startswith('seg_layers') and not k.startswith('outs')}
        # for k in ['fed_task', 'patches', 'deep_supervision', 'condition_on_seg', 'output_seg']:
        #     if k in ckpt:
        #         del ckpt[k]
        ckpt = self.get_ckpt(self.hps_ckpt_fname, 'hps')
        model.load_state_dict(ckpt)
        heart_roi = sitk.ReadImage(self.heart_roi_fname)
        heart_seg = sitk.ReadImage(self.heart_seg_fname)
        x_torch, heart_roi_res, heart_seg_res = self.transform(heart_roi, heart_seg)
        if self.model_type == 'nnunet':
            logits = model(x_torch.to(self.device))[0]
        elif self.model_type == 'swin_unetr':
            with torch.cuda.amp.autocast():
                logits = sliding_window_inference(x_torch.to(self.device), (96,96,96), 4, model)
        # logits = [model(x_torch.to(self.device))[0].cpu() for _ in range(5)]
        # logits = [torch.softmax(l, dim=1) for l in logits]
        # logits = torch.stack(logits)
        # # logits = logits.mean(axis=0)
        # logits, uncert, ale, epi = self.uncertainty(logits, var=None)
        # pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)[0].cpu()
        pred = torch.argmax(logits, dim=1)[0].cpu()
        pts_idx_pred, pts_pred = self.pts_from_pred(pred, heart_roi_res)
        if len(pts_pred) == 5:
            self.hps_fname = self.tmp_dir / 'hps.mps'
            mps_from_pts(np.array(pts_pred), self.hps_fname)
        
        # self.save_preds_and_uncert(logits, uncert, ale, epi, name='HPS', ref_sitk=heart_roi_res)
        return pts_pred

    @torch.no_grad()
    def membranous_septum(self):
        if self.model_type == 'nnunet':
            model = nnunet_configuration(num_segmentation_heads=3, num_input_channels=2).to(self.device)
        elif self.model_type == 'swin_unetr':
            model = SwinUNETR(
                img_size=(96,96,96),
                in_channels=2,
                out_channels=3,
                feature_size=48,
                use_checkpoint=False,
            ).to(self.device)
        model.eval()
        # ckpt = torch.load(self.ms_ckpt_fname)
        ckpt = self.get_ckpt(self.ms_ckpt_fname, 'ms')
        # ckpt = {k.replace('unet.', ''): v for k, v in ckpt.items()}
        # del ckpt['fed_task']
        model.load_state_dict(ckpt)
        heart_roi = sitk.ReadImage(self.heart_roi_fname)
        heart_seg = sitk.ReadImage(self.heart_seg_fname)
        x_torch, heart_roi_res, heart_seg_res = self.transform(heart_roi, heart_seg)
        if self.model_type == 'nnunet':
            logits = model(x_torch.to(self.device))[0]
        elif self.model_type == 'swin_unetr':
            with torch.cuda.amp.autocast():
                logits = sliding_window_inference(x_torch.to(self.device), (96,96,96), 4, model)
        # logits = [nnunet(x_torch.to(self.device))[0].cpu() for _ in range(5)]
        # logits = [torch.softmax(l, dim=1) for l in logits]
        # logits = torch.stack(logits)
        # # logits = logits.mean(axis=0)
        # logits, uncert, ale, epi = self.uncertainty(logits, var=None)
        # pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)[0].cpu()
        pred = torch.argmax(logits, dim=1)[0].cpu()
        pts_idx_pred, pts_pred = self.pts_from_pred(pred, heart_roi_res)
        if len(pts_pred) == 2:
            self.ms_fname = self.tmp_dir / 'ms.mps'
            mps_from_pts(np.array(pts_pred), self.ms_fname)
        
        # self.save_preds_and_uncert(logits, uncert, ale, epi, name='MS', ref_sitk=heart_roi_res)
        return pts_pred

    @torch.no_grad()
    def calcification(self):
        if self.model_type == 'nnunet':
            model = nnunet_configuration(num_segmentation_heads=2, num_input_channels=2).to(self.device)
        elif self.model_type == 'swin_unetr':
            model = SwinUNETR(
                img_size=(96,96,96),
                in_channels=2,
                out_channels=2,
                feature_size=48,
                use_checkpoint=False,
            ).to(self.device)
        model.eval()
        # nnunet = nnunet_configuration(num_segmentation_heads=2, num_input_channels=2).to(self.device)
        # ckpt = torch.load(self.calc_ckpt_fname)
        # ckpt = {k.replace('unet.', ''): v for k, v in ckpt.items()}
        # del ckpt['fed_task']
        ckpt = self.get_ckpt(self.calc_ckpt_fname, 'calc')
        model.load_state_dict(ckpt)
        if self.heart_seg_fname is None:
            self.segment_heart()
        heart_roi = sitk.ReadImage(self.heart_roi_fname)
        heart_seg = sitk.ReadImage(self.heart_seg_fname)
        x_torch, heart_roi_res, heart_seg_res = self.transform(heart_roi, heart_seg)
        if self.model_type == 'nnunet':
            logits = model(x_torch.to(self.device))[0]
        elif self.model_type == 'swin_unetr':
            with torch.cuda.amp.autocast():
                logits = sliding_window_inference(x_torch.to(self.device), (96,96,96), 4, model)
        # logits = [nnunet(x_torch.to(self.device))[0].cpu() for _ in range(5)]
        # logits = [torch.softmax(l, dim=1) for l in logits]
        # logits = torch.stack(logits)
        # # logits = logits.mean(axis=0)
        # logits, uncert, ale, epi = self.uncertainty(logits, var=None)
        # pred = torch.argmax(torch.softmax(logits, dim=1), dim=1)[0].cpu()
        pred = torch.argmax(logits, dim=1)[0].cpu()
        pred_sitk = self.torch_to_sitk(pred, heart_roi_res)
        self.calc_fname = self.tmp_dir / 'calc.nii.gz'
        sitk.WriteImage(pred_sitk, self.calc_fname)

        # self.save_preds_and_uncert(logits, uncert, ale, epi, name='Calc', ref_sitk=heart_roi_res)
    
    # @staticmethod 
    # def uncertainty(p_hat, var='sum'):
    #     p_mean = torch.mean(p_hat, dim=0)
    #     ale = torch.mean(p_hat*(1-p_hat), dim=0)
    #     epi = torch.mean(p_hat**2, dim=0) - p_mean**2
    #     if var == 'sum':
    #         ale = torch.sum(ale, dim=0)
    #         epi = torch.sum(epi, dim=0)
    #     elif var == 'top':
    #         ale = ale[torch.argmax(p_mean)]
    #         epi = epi[torch.argmax(p_mean)]
    #     uncert = ale + epi
    #     return p_mean, uncert, ale, epi
    
    # def save_preds_and_uncert(self, logits, uncert, ale, epi, name, ref_sitk):
    #     logits, uncert, ale, epi = logits[0], uncert[0], ale[0], epi[0]
    #     for x, n in zip([logits, uncert, ale, epi], ['Logits', 'Uncertainty']): # , 'Aleatoric', 'Epistemic']):
    #         (self.tmp_dir / f'{n} {name} Torch').mkdir(exist_ok=True)
    #         torch_fname = f'{n.lower()}_{name.lower()}_torch_fname'
    #         setattr(self, torch_fname, self.tmp_dir / f'{n} {name} Torch' / f'{n.lower()}_{name.lower()}.pt')
    #         torch.save(x, getattr(self, torch_fname))

    #         # sum over logits might not be the best idea, but only for visualization purposes
    #         x_sitk = self.torch_to_sitk(x.sum(dim=0), ref_sitk)
    #         (self.tmp_dir / f'{n} {name}').mkdir(exist_ok=True)
    #         fname = f'{n.lower()}_{name}_fname'
    #         setattr(self, fname, self.tmp_dir / f'{n} {name}' / f'{n.lower()}_{name.lower()}.nii.gz')
    #         sitk.WriteImage(x_sitk, getattr(self, fname))
    
    # def extract_aorta_seg_measures(self, segmentation, spacing):
    #     contours, _ = cv2.findContours(segmentation.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     if not len(contours):
    #         return [0]*6
    #     contour_lengths = [len(c) for c in contours]
    #     contour = contours[np.argmax(contour_lengths)]
    #     if len(contour) < 5:
    #         return [0]*6
        
    #     # seg_rgb = cv2.cvtColor(segmentation, cv2.COLOR_GRAY2RGB)
    #     # cv2.drawContours(seg_rgb, contour, -1, (1,0,0), 1)
    #     # cv2.imwrite('tmp/aorta_seg.png', seg_rgb*255)

    #     perimeter = cv2.arcLength(contour, True)
    #     area = cv2.contourArea(contour)
    #     area_derived_diameter = 2 * np.sqrt(area / np.pi)

    #     average_spacing = (spacing[0] + spacing[1]) / 2
    #     perimeter_in_mm = perimeter * average_spacing
    #     perimeter_derived_diameter_in_mm = perimeter_in_mm / np.pi
    #     area_in_mm = area * spacing[0] * spacing[1]
    #     area_derived_diameter_in_mm = 2 * np.sqrt(area_in_mm / np.pi)

    #     ellipse = cv2.fitEllipse(contour)
    #     (center_x, center_y), axes, angle = ellipse

    #     major_axis, minor_axis = np.max(axes), np.min(axes)

    #     angle_rad = np.deg2rad(angle)
    #     cos_angle = np.cos(angle_rad)
    #     sin_angle = np.sin(angle_rad)

    #     major_axis_x_component = major_axis * cos_angle
    #     major_axis_y_component = major_axis * sin_angle
    #     major_axis_in_mm = np.sqrt((major_axis_x_component * spacing[0])**2 + (major_axis_y_component * spacing[1])**2)

    #     minor_axis_x_component = minor_axis * (-sin_angle)
    #     minor_axis_y_component = minor_axis * cos_angle
    #     minor_axis_in_mm = np.sqrt((minor_axis_x_component * spacing[0])**2 + (minor_axis_y_component * spacing[1])**2)

    #     return perimeter_derived_diameter_in_mm, perimeter_in_mm, area_derived_diameter_in_mm, area_in_mm, major_axis_in_mm, minor_axis_in_mm
    
    # def tavi_parameters(self, tavi_params_fname=None):
    #     hps = pts_from_mps(self.hps_fname)
    #     if len(hps) < 5:
    #         return {}
    #     ms = pts_from_mps(self.ms_fname)
    #     if len(ms) < 2:
    #         return {}
    #     calc = sitk.ReadImage(self.calc_fname)
    #     # aorta_seg = sitk.ReadImage(self.aorta_seg_fname)

    #     # AA Plane diameter -> ellipses major and minor axes
    #     heart_seg = sitk.ReadImage(self.heart_seg_fname)
    #     heart_seg_foc = focus_valve_region_from_pts(heart_seg, pts=hps, step_mm=40)
    #     heart_seg_foc_np = sitk.GetArrayFromImage(heart_seg_foc)
    #     heart_seg_foc_np[heart_seg_foc_np == 3] = 6 # Merge LV with Aorta
    #     heart_seg_foc_np[heart_seg_foc_np != 6] = 0 # Set everything else to 0
    #     heart_seg_foc_new = sitk.GetImageFromArray(heart_seg_foc_np)
    #     heart_seg_foc_new.CopyInformation(heart_seg_foc)
    #     sitk.WriteImage(heart_seg_foc_new, self.tmp_dir / 'aorta_and_lv_seg_foc.nii.gz')

    #     point = np.array([0,0,0])
    #     _, normal = plane_from_three_points(hps[:3])
    #     seg_reader = vtk.vtkNIFTIImageReader()
    #     seg_reader.SetFileName(self.tmp_dir / 'aorta_and_lv_seg_foc.nii.gz') # self.aorta_seg_fname)
    #     seg_reader.Update()
    #     seg_data = seg_reader.GetOutput()
        
    #     AAP_slice, AAP_slice_spacing = self.extract_slice_from_normal_vector_and_point(seg_data, normal, point)
    #     # aa_plane_size_mm_sq = np.sum(aa_plane) * spacing[0] * spacing[1]

    #     measure_labels = ['D_peri_mm', 'peri_mm', 'D_area_mm', 'area_mm', 'maj_axis_mm', 'min_axis_mm']
    #     measures_to_dict = lambda m, prefix: {f'{prefix}_{l}': x for l, x in zip(measure_labels, m)}

    #     AAP_measures = self.extract_aorta_seg_measures(AAP_slice, AAP_slice_spacing)

    #     # AORTA PARAMS
    #     aorta_img_slices, aorta_seg_slices, aorta_slices_spacings, aorta_centerline_pts = self.aorta_slices_from_hps_and_centerline()
    #     aorta_measures_perpendicular_to_centerline = [
    #         self.extract_aorta_seg_measures(sl, sp) for sl, sp in zip(aorta_seg_slices, aorta_slices_spacings)
    #     ]

    #     # STJ
    #     aorta_areas_perpendicular_to_centerline = np.array([m[3] for m in aorta_measures_perpendicular_to_centerline])
    #     aorta_areas = aorta_areas_perpendicular_to_centerline[aorta_areas_perpendicular_to_centerline > 0]
    #     x = np.arange(len(aorta_areas))
    #     _stj_loc, stj_aorta_size_mm_sq, x_func, y_func = self.get_minimum_of_aorta_sizes(x, aorta_areas, degree=7)
    #     stj_loc = int(np.round(_stj_loc)) + len(aorta_measures_perpendicular_to_centerline) - len(aorta_areas)
    #     stj_measures_perpendicular_to_centerline = aorta_measures_perpendicular_to_centerline[stj_loc]
        
    #     fig, ax = plt.subplots()
        
    #     cp = sns.color_palette()
    #     ax.scatter(x, aorta_areas, color=cp[0], marker='x', label='Size of Aorta')
    #     ax.plot(x_func, y_func, color=cp[1], label='Polynom')
    #     ax.axvline(_stj_loc, color='black', label='STJ Diam.')
    #     ax.set_xlabel(r'Loc. on Aorta Centerline (0 = Middle of HPs) [mm]')
    #     ax.set_ylabel(r'$A_{\mathrm{Aorta}}\,[mm^{2}]$')
    #     ax.legend(loc='lower right')
    #     fig.tight_layout()
    #     fig.savefig(self.tmp_dir / 'aorta_radius.png', bbox_inches='tight')
    #     plt.close(fig)
        
    #     # distance RCA and LCA to AA Plane
    #     distance_to_AAP = lambda x: np.abs(np.dot(normal, x)) / np.linalg.norm(normal)
    #     rca, lca = hps[3], hps[4]
    #     D_rca = distance_to_AAP(rca)
    #     D_lca = distance_to_AAP(lca)
        
    #     # SOV
    #     D_centerline_pts = [distance_to_AAP(x) for x in aorta_centerline_pts]
    #     AAP_idx = np.argmin(D_centerline_pts)
    #     min_d, max_d = np.min([D_rca, D_lca]), np.max([D_rca, D_lca])
    #     mean_d = np.mean([D_rca, D_lca])
    #     diff_d_mean_D_centerline_pts = np.abs(np.array([d - mean_d for d in D_centerline_pts]))
    #     sov_idx = np.argmin(diff_d_mean_D_centerline_pts)
    #     # idxs_between_cas = np.where((D_centerline_pts[AAP_idx:] >= min_d) & (D_centerline_pts[AAP_idx:] <= max_d))[0]
    #     # sov_idx = idxs_between_cas[int(len(idxs_between_cas) / 2)]
    #     try:
    #         sov_measures_perpendicular_to_centerline = aorta_measures_perpendicular_to_centerline[sov_idx + AAP_idx]
    #     except:
    #         return {}

    #     # distance MS2 to AA plane
    #     ms2 = ms[1]
    #     D_ms2 = distance_to_AAP(ms2)
        
    #     # Calc per cusp

    #     ## Find all calc volumes
    #     import cc3d
    #     from scipy import ndimage
    #     calc_np = sitk.GetArrayFromImage(calc)
    #     labels = cc3d.connected_components(calc_np)
    #     sizes = np.bincount(labels.flatten())[1:]
        
    #     ## Find closest cusp
    #     centers_px = ndimage.center_of_mass(labels, labels, np.arange(len(sizes))+1)
    #     centers_mm = np.array([calc.TransformContinuousIndexToPhysicalPoint(pt[::-1]) for pt in centers_px])
    #     distances = np.array([[np.linalg.norm(c-pt) for pt in hps] for c in centers_mm])
    #     min_distances = np.array([np.min(d[:3]) for d in distances])
    #     vols_to_cusp = np.array([np.argmin(d) for d in distances])

    #     _unique_labels = np.unique(labels)[1:]
    #     sizes, vols_to_cusp = sizes[min_distances < 20], vols_to_cusp[min_distances < 20]
    #     _unique_labels = _unique_labels[min_distances < 20]
    #     calc_to_cusp_seg = np.zeros_like(calc_np)
    #     calcification = np.zeros((5,))
    #     voxel_size = np.prod(calc.GetSpacing())
    #     for s, cusp_idx, _l in zip(sizes, vols_to_cusp, _unique_labels):
    #         calcification[cusp_idx] += s * voxel_size
    #         calc_to_cusp_seg[labels == _l] = cusp_idx + 1
    #     calc_to_cusp_seg = sitk.GetImageFromArray(calc_to_cusp_seg)
    #     calc_to_cusp_seg.CopyInformation(calc)
    #     sitk.WriteImage(calc_to_cusp_seg, self.tmp_dir / 'calc_to_cusp_seg.nii.gz')
    #     # sitk.WriteImage(sitk.ReadImage(self.heart_roi_fname), 'tmp/heart_roi.nii.gz')
    #     # sitk.WriteImage(calc, 'tmp/calc.nii.gz')

    #     AAP_measures = measures_to_dict(AAP_measures, 'AAP')
    #     STJ_measures = measures_to_dict(stj_measures_perpendicular_to_centerline, 'STJ')
    #     SOV_measures = measures_to_dict(sov_measures_perpendicular_to_centerline, 'SOV')

    #     hps, ms = hps.tolist(), ms.tolist()
    #     tavi_params = {
    #         'RCC': hps[0], 'LCC': hps[1], 'ACC': hps[2], 'RCA': hps[3], 'LCA': hps[4],
    #         'MS1': ms[0], 'MS2': ms[1],
    #         'Distance_AAP_RCA': D_rca, 'Distance_AAP_LCA': D_lca, 'Distance_AAP_ms2': D_ms2,
    #         'Calc_RCC': calcification[0], 'Calc_LCC': calcification[1], 'Calc_ACC': calcification[2],
    #         'Calc_RCA': calcification[3], 'Calc_LCA': calcification[4],
    #         **AAP_measures, **STJ_measures, **SOV_measures
    #     }
    #     if tavi_params_fname is not None:
    #         with open(self.tmp_dir / tavi_params_fname, 'w') as f:
    #             json.dump(tavi_params, f)
    #     return tavi_params

    # @staticmethod
    # def get_minimum_of_aorta_sizes(x, aorta_sizes_along_centerline, degree=7):
    #     y = np.array(aorta_sizes_along_centerline)
    #     # y = y[y > 0]
    #     # x = np.arange(len(y))
    #     coefficients = np.polyfit(x, y, degree)
    #     poly_func = np.poly1d(coefficients)
    #     x_range = np.linspace(min(x), max(x), 100)
    #     y_fit = poly_func(x_range)

    #     derivative_coefficients = np.polyder(coefficients)
    #     derivative_func = np.poly1d(derivative_coefficients)
    #     y_derivative = derivative_func(x_range)

    #     second_derivative_coefficients = np.polyder(derivative_coefficients)
    #     second_derivative_func = np.poly1d(second_derivative_coefficients)

    #     # Find x-values where first derivative is zero
    #     roots = fsolve(derivative_func, np.linspace(min(x), max(x), degree))  # Using multiple starting points

    #     # Filter the roots to be within the range of interest
    #     roots = [root for root in roots if min(x) <= root <= max(x)]

    #     minima = []
    #     maxima = []

    #     for root in roots:
    #         # Use the second derivative test
    #         if second_derivative_func(root) > 0:
    #             minima.append((root, poly_func(root)))
    #         elif second_derivative_func(root) < 0:
    #             maxima.append((root, poly_func(root)))
    #     # mins = [m for m in minima if m[0] > 0.0 and m[0] < 0.4 and m[1] > 500]
    #     # mins = [m for m in minima if m[0] > 25 and m[0] < 90 and m[1] > 400]
    #     # mins = [m for m in minima if m[0] < 80 and m[1] > 50]
    #     mins = minima
    #     if len(mins):
    #         min_idx, min_val = mins[0]
    #     else:
    #         print(minima)
    #         min_idx, min_val = 30, 700

    #     return min_idx, min_val, x_range, y_fit

    def transform(
            self, 
            heart_roi, 
            heart_seg, 
            target_resolution=(192,192,192),
            target_spacing=(1,1,1)
        ):
        # intensityproperties = {
        #     "max": 2176.0,
        #     "mean": 373.7686767578125,
        #     "median": 352.0,
        #     "min": -813.0,
        #     "percentile_00_5": -42.0,
        #     "percentile_99_5": 878.0,
        #     "std": 166.8054656982422
        # }
        ct_normalization = CTNormalization(intensityproperties=self.intensityproperties)
        heart_roi_norm = ct_normalization.run(torch.from_numpy(sitk.GetArrayFromImage(heart_roi).astype(np.float32)))
        heart_roi_norm =sitk.GetImageFromArray(heart_roi_norm.numpy())
        heart_roi_norm.CopyInformation(heart_roi)
        heart_roi_norm = resample_image(heart_roi_norm, out_spacing=target_spacing)
        heart_roi_norm = crop_or_pad_img(heart_roi_norm, target_resolution)
        heart_roi_torch = self.sitk_to_torch(heart_roi_norm).to(self.device)
        heart_seg = resample_image(heart_seg, out_spacing=target_spacing, interpolator='NearestNeighbor')
        heart_seg = crop_or_pad_seg(heart_seg, target_resolution)
        heart_seg_torch = self.sitk_to_torch(heart_seg).to(self.device)
        x_torch = torch.cat([heart_roi_torch, heart_seg_torch], dim=0)
        
        return x_torch[None], heart_roi_norm, heart_seg

    @staticmethod
    def pts_from_pred(pred, ref_sitk):
        if type(pred) == torch.Tensor:
            pred = pred.cpu().detach().numpy()
        largest_vols = find_largest_volume(pred)
        pts_idx_pred = find_center_of_mass(largest_vols).tolist()
        pts_pred = np.array([ref_sitk.TransformContinuousIndexToPhysicalPoint(pt_idx[::-1]) for pt_idx in pts_idx_pred])
        return pts_idx_pred, pts_pred

    @staticmethod
    def sitk_to_torch(img_sitk):
        arr = sitk.GetArrayFromImage(img_sitk)
        # arr = np.einsum('zyx->xyz', arr)
        return torch.from_numpy(arr.astype(np.float32))[None]

    @staticmethod
    def torch_to_sitk(tensor, ref_sitk):
        # arr = np.einsum('xyz->zyx', tensor.detach().cpu().numpy().astype(np.float32))
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        img = sitk.GetImageFromArray(arr)
        img.CopyInformation(ref_sitk)
        return img
    
    # def set_origin_to_center_of_hinge_points(self):
    #     hps = pts_from_mps(self.hps_fname)
    #     if len(hps) < 5:
    #         return False
    #     ms = pts_from_mps(self.ms_fname)
    #     heart_roi = sitk.ReadImage(self.heart_roi_fname)
    #     heart_seg = sitk.ReadImage(self.heart_seg_fname)
    #     calc = sitk.ReadImage(self.calc_fname)
    #     calc_origin_difference = np.array(heart_roi.GetOrigin()) - np.array(calc.GetOrigin())

    #     hps_idx = [heart_roi.TransformPhysicalPointToContinuousIndex(pt.tolist()) for pt in hps]
    #     ms_idx = [heart_roi.TransformPhysicalPointToContinuousIndex(pt.tolist()) for pt in ms]
        
    #     center_pt, radius = define_circle_3d(*hps[:3])
    #     center_pt_idx = heart_roi.TransformPhysicalPointToContinuousIndex(center_pt.tolist())
    #     heart_roi.SetOrigin((0,0,0))
    #     center_pt_new = heart_roi.TransformContinuousIndexToPhysicalPoint(center_pt_idx)
    #     new_origin = [-i for i in center_pt_new] 
    #     heart_roi.SetOrigin(new_origin)
    #     heart_seg.SetOrigin(new_origin)
    #     new_calc_origin = new_origin - calc_origin_difference
    #     calc.SetOrigin(new_calc_origin)
    #     self.heart_roi_fname = self.tmp_dir / 'Heart ROI New Origin' / 'heart_roi_new_origin.nii.gz'
    #     self.heart_seg_fname = self.tmp_dir / 'Heart Seg New Origin' / 'heart_seg_new_origin.nii.gz'
    #     self.calc_fname = self.tmp_dir / 'Calc New Origin' / 'calc_new_origin.nii.gz'
    #     self.heart_roi_fname.parents[0].mkdir(exist_ok=True)
    #     self.heart_seg_fname.parents[0].mkdir(exist_ok=True)
    #     self.calc_fname.parents[0].mkdir(exist_ok=True)
    #     sitk.WriteImage(heart_roi, self.heart_roi_fname)
    #     sitk.WriteImage(heart_seg, self.heart_seg_fname)
    #     sitk.WriteImage(calc, self.calc_fname)

    #     hps_new_origin = np.array([heart_roi.TransformContinuousIndexToPhysicalPoint(pt) for pt in hps_idx])
    #     ms_new_origin = np.array([heart_roi.TransformContinuousIndexToPhysicalPoint(pt) for pt in ms_idx])
    #     self.hps_fname = self.tmp_dir / 'HPS New Origin' / 'hps_new_origin.mps'
    #     self.ms_fname = self.tmp_dir / 'MS New Origin' / 'ms_new_origin.mps'
    #     self.hps_fname.parents[0].mkdir(exist_ok=True)
    #     self.ms_fname.parents[0].mkdir(exist_ok=True)
    #     mps_from_pts(hps_new_origin, self.hps_fname)
    #     mps_from_pts(ms_new_origin, self.ms_fname)
    #     return True
    
    # @staticmethod
    # def fit_quadratic_polynom(x, y, z, normal_in_origin):
    #     from scipy.optimize import minimize
    #     def quadratic_polynomial(coefs, x, y):
    #         a, b, c, d, e, f = coefs
    #         return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

    #     def objective_function(coefs, x, y, z):
    #         return np.sum((quadratic_polynomial(coefs, x, y) - z)**2)

    #     def constraint1(coefs, x0, y0, u):
    #         a, b, c, d, e, f = coefs
    #         return 2*a*x0 + c*y0 + d - u

    #     def constraint2(coefs, x0, y0, v):
    #         a, b, c, d, e, f = coefs
    #         return 2*b*y0 + c*x0 + e - v
        
    #     normal_in_origin /= normal_in_origin[2]
    #     u, v, w = normal_in_origin
    #     x0 = 0
    #     y0 = 0

    #     initial_guess = np.array([1, 1, 1, 1, 1, 1])

    #     con1 = {'type': 'eq', 'fun': constraint1, 'args': (x0, y0, u)}
    #     con2 = {'type': 'eq', 'fun': constraint2, 'args': (x0, y0, v)}
    #     constraints = [con1, con2]

    #     res = minimize(objective_function, initial_guess, args=(x, y, z), constraints=constraints)

    #     x = np.linspace(x.min(), x.max(), 100)
    #     y = np.linspace(y.min(), y.max(), 100)

    #     z = quadratic_polynomial(res.x, x, y)
    #     pts = np.concatenate([x[::-1,None], y[::-1,None], z[::-1,None]], axis=1)
    #     return pts
    
    # @staticmethod
    # def fit_cubic_polynom(x, y, z, normal_in_origin, additional_constraints=None):
    #     from scipy.optimize import minimize
    #     def cubic_polynomial(coefs, x, y):
    #         a, b, c, d, e, f, g, h, i, j = coefs
    #         return a*x**3 + b*y**3 + c*x**2*y + d*x*y**2 + e*x**2 + f*y**2 + g*x*y + h*x + i*y + j

    #     def objective_function(coefs, x, y, z):
    #         return np.sum((cubic_polynomial(coefs, x, y) - z)**2)

    #     def constraint1(coefs, x0, y0, u):  # partial x derivative
    #         a, b, c, d, e, f, g, h, i, j = coefs
    #         return 3*a*x0**2 + c*y0**2 + 2*e*x0 + d*y0 + g*y0 + h - u

    #     def constraint2(coefs, x0, y0, v):  # partial y derivative
    #         a, b, c, d, e, f, g, h, i, j = coefs
    #         return 3*b*y0**2 + c*x0**2 + 2*f*y0 + d*x0 + g*x0 + i - v

    #     normal_in_origin /= normal_in_origin[2]
    #     u, v, w = normal_in_origin
    #     x0 = 0
    #     y0 = 0

    #     initial_guess = np.array([1]*10)  # 10 coefficients for the cubic polynomial

    #     con1 = {'type': 'eq', 'fun': constraint1, 'args': (x0, y0, u)}
    #     con2 = {'type': 'eq', 'fun': constraint2, 'args': (x0, y0, v)}
    #     constraints = [con1, con2]
    #     if additional_constraints is not None:
    #         constraints += [
    #             {'type': 'eq', 'fun': constraint1, 'args': additional_constraints[0]},
    #             {'type': 'eq', 'fun': constraint1, 'args': additional_constraints[1]},
    #         ]

    #     res = minimize(objective_function, initial_guess, args=(x, y, z), constraints=constraints)

    #     x = np.linspace(x.min(), x.max(), 100)
    #     y = np.linspace(y.min(), y.max(), 100)

    #     z = cubic_polynomial(res.x, x, y)
    #     pts = np.concatenate([x[::-1,None], y[::-1,None], z[::-1,None]], axis=1)
    #     return pts

    # @staticmethod
    # def fit_quartic_polynom(x, y, z, normal_in_origin):
    #     from scipy.optimize import minimize
    #     def quartic_polynomial(coefs, x, y):
    #         a, b, c, d, e, f, g, h, i, j, k, l, m, n, o = coefs
    #         return a*x**4 + b*y**4 + c*x**3*y + d*x*y**3 + e*x**3 + f*y**3 + g*x**2*y**2 + h*x**2*y + i*x*y**2 + j*x**2 + k*y**2 + l*x*y + m*x + n*y + o

    #     def objective_function(coefs, x, y, z):
    #         return np.sum((quartic_polynomial(coefs, x, y) - z)**2)

    #     def constraint1(coefs, x0, y0, u):  # partial x derivative
    #         a, b, c, d, e, f, g, h, i, j, k, l, m, n, o = coefs
    #         return 4*a*x0**3 + 3*c*x0**2*y0 + d*y0**3 + 3*e*x0**2 + 2*g*x0*y0**2 + 2*h*x0*y0 + i*y0**2 + 2*j*x0 + l*y0 + m - u

    #     def constraint2(coefs, x0, y0, v):  # partial y derivative
    #         a, b, c, d, e, f, g, h, i, j, k, l, m, n, o = coefs
    #         return 4*b*y0**3 + c*x0**3 + 3*d*x0*y0**2 + 3*f*y0**2 + 2*g*x0**2*y0 + h*x0**2 + 2*i*x0*y0 + 2*k*y0 + l*x0 + n - v

    #     normal_in_origin /= normal_in_origin[2]
    #     u, v, w = normal_in_origin
    #     x0 = 0
    #     y0 = 0

    #     initial_guess = np.array([1]*15)  # 15 coefficients for the quartic polynomial

    #     con1 = {'type': 'eq', 'fun': constraint1, 'args': (x0, y0, u)}
    #     con2 = {'type': 'eq', 'fun': constraint2, 'args': (x0, y0, v)}
    #     constraints = [con1, con2]

    #     res = minimize(objective_function, initial_guess, args=(x, y, z), constraints=constraints)

    #     x = np.linspace(x.min(), x.max(), 100)
    #     y = np.linspace(y.min(), y.max(), 100)

    #     z = quartic_polynomial(res.x, x, y)
    #     pts = np.concatenate([x[::-1,None], y[::-1,None], z[::-1,None]], axis=1)
    #     return pts


    # def aorta_centerline(self): #, slicer_path='/mnt/ssd/applications/Slicer-5.2.2-linux-amd64/Slicer'):
    #     if self.hps_fname is None:
    #         self.hinge_points()
    #     hps = pts_from_mps(self.hps_fname)
    #     heart_seg = sitk.ReadImage(self.heart_seg_fname)
    #     heart_roi = sitk.ReadImage(self.heart_roi_fname)
    #     heart_roi_foc = focus_valve_region_from_pts(heart_roi, pts=hps, step_mm=40)
    #     heart_seg_foc = focus_valve_region_from_pts(heart_seg, pts=hps, step_mm=40)
    #     # sitk.WriteImage(heart_roi_foc, self.tmp_dir / 'heart_roi_foc.nii.gz')
    #     heart_seg_np = sitk.GetArrayFromImage(heart_seg_foc)
    #     aorta_seg_np = (heart_seg_np == 6).astype(np.uint8)
    #     aorta_seg_np = find_largest_volume(aorta_seg_np).astype(np.uint8)
    #     aorta_seg = sitk.GetImageFromArray(aorta_seg_np)
    #     aorta_seg.SetSpacing(heart_seg_foc.GetSpacing())
    #     # aorta_seg.CopyInformation(heart_seg_foc)
    #     hps_idx = [heart_seg_foc.TransformPhysicalPointToContinuousIndex(pt.tolist()) for pt in hps]
    #     hps_aorta = np.array([aorta_seg.TransformContinuousIndexToPhysicalPoint(pt) for pt in hps_idx])
    #     # mps_from_pts(hps_aorta, self.tmp_dir / 'aorta_hps.mps')
    #     point, normal = plane_from_three_points(hps_aorta[:3])
    #     center_pt, radius = define_circle_3d(*hps_aorta[:3])
    #     start_point_centerline = center_pt - 1 * normal
    #     self.aorta_seg_fname = self.tmp_dir / 'tavi_aorta_seg.nii.gz'
    #     sitk.WriteImage(aorta_seg, self.tmp_dir / 'tavi_aorta_seg.nii.gz')
    #     self.heart_roi_foc_fname = self.tmp_dir / 'tavi_heart_roi_foc.nii.gz'
    #     sitk.WriteImage(heart_roi_foc, self.tmp_dir / 'tavi_heart_roi_foc.nii.gz')
    #     start_point_fname = self.tmp_dir / 'tavi_start_point_centerline.npy'
    #     with open(start_point_fname, 'wb') as f:
    #         np.save(f, start_point_centerline)
    #     origin_fname = self.tmp_dir / 'tavi_origin_centerline.npy'
    #     with open(origin_fname, 'wb') as f:
    #         np.save(f, np.array(heart_seg_foc.GetOrigin()))
        
    #     client = docker.from_env()
        
    #     if Path('/.dockerenv').exists():
    #         import socket
    #         short_id = socket.gethostname()
    #         this_container = [c for c in client.containers.list() if c.id[:len(short_id)] == short_id][0]
    #         host_tmp_dir = [m for m in this_container.attrs['Mounts'] if m['Destination'] != '/var/run/docker.sock'][0]['Source']
    #     else:   
    #         host_tmp_dir = str(self.tmp_dir)

    #     slicer_container = client.containers.run(
    #         'maltetoelle/slicer-jupyter-vmtk:latest',
    #         name='tavi-slicer-vmtk',
    #         volumes=[f'{host_tmp_dir}:/home/sliceruser/data'],
    #         detach=True
    #     )
    #     slicer_executable = '/home/sliceruser/Slicer/Slicer'
    #     flags = '--no-main-window --python-script'
    #     path_to_script = '/home/sliceruser/slicer_centerline_extraction.py'
    #     data_dir = '/home/sliceruser/data'
    #     path_to_seg = f'{data_dir}/tavi_aorta_seg.nii.gz'
    #     path_to_start_point = f'{data_dir}/tavi_start_point_centerline.npy'
    #     path_to_origin = f'{data_dir}/tavi_origin_centerline.npy'
        
    #     slicer_container.exec_run(f'{slicer_executable} {flags} {path_to_script} {path_to_seg} {data_dir} {path_to_start_point} {path_to_origin}')
    #     slicer_container.stop()
    #     slicer_container.remove()
    #     # centerline_script_path = '/mnt/ssd/git-repos/floto-tavi-outcome-prediction/slicer_centerline_extraction.py'
    #     # command = f'{slicer_path} --no-main-window --python-script "{centerline_script_path}" "{str(self.tmp_dir.absolute())}/tavi_aorta_seg.nii.gz" "{str(self.tmp_dir.absolute())}" "{str(start_point_fname.absolute())}" "{str(origin_fname.absolute())}" > /dev/null 2>&1'
    #     # print(command)
    #     # print(start_point_centerline)
    #     # os.system(command)
    #     self.centerline_pts_fname = self.tmp_dir / 'centerline.mps'

    #     centerline_pts = pts_from_mps(self.centerline_pts_fname)[::-1]#[3:]
    #     # (self.tmp_dir / 'centerline.mps').rename(self.tmp_dir / 'centerline_slicer.mps')

    #     if not len(centerline_pts):
    #         return

    #     center_pt = np.array([0,0,0])
    #     # start_centerline_pt = centerline_pts[0]
    #     # trajectory = start_centerline_pt - center_pt
    #     # trajectory_pts = np.array([(center_pt + i * trajectory).tolist() for i in np.linspace(0,.9,30)])
    #     trajectory_pts = np.array([(center_pt + i * normal).tolist() for i in range(-8,10)])

    #     # normal_centerline_start = centerline_pts[0:4]
    #     # normal_centerline_start = normal_centerline_start.mean(axis=0)
    #     # u, v, w = normal_centerline_start[0]
    #     # x0, y0, _ = centerline_pts[0]
    #     # centerline_constraints = [(x0, y0, u), (x0, y0, v)]
    #     point1 = trajectory_pts[-1]
    #     point2 = centerline_pts[0]

    #     num_interpolated_points = 10
    #     interpolated_points = np.array([
    #         (1 - t) * point1 + t * point2 for t in np.linspace(0, 1, num_interpolated_points + 2)[1:-1]
    #     ])
    #     centerline_pts = np.concatenate([trajectory_pts, interpolated_points, centerline_pts])

    #     # x, y, z = [np.concatenate([trajectory_pts[:,i], centerline_pts[:,i]], axis=0) for i in range(3)]

    #     # centerline_pts_reg = self.fit_cubic_polynom(x, y, z, -normal)

    #     # np.savez(self.tmp_dir / 'centerline_and_traj_pts.npz', centerline_pts_slicer=centerline_pts, trajectory_pts=trajectory_pts, centerline_pts_reg=centerline_pts_reg)

    #     # from scipy.interpolate import CubicSpline
    #     from scipy.interpolate import UnivariateSpline as Spline
    #     points = centerline_pts
    #     distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    #     distances = np.insert(distances, 0, 0)  # First point is at a distance 0

    #     # Create cubic spline data for each dimension.
    #     cs_x = Spline(distances, points[:, 0], s=5.)
    #     cs_y = Spline(distances, points[:, 1], s=5.)
    #     cs_z = Spline(distances, points[:, 2], s=5.)

    #     # Create a fine array of "distances" to evaluate the spline and plot it.
    #     num_points_fine = 100  # For example, 1000 points for a smooth curve
    #     distance_fine = np.linspace(distances[0], distances[-1], num_points_fine)

    #     centerline_pts = np.vstack([cs_x(distance_fine), cs_y(distance_fine), cs_z(distance_fine)]).T

    #     mps_from_pts(centerline_pts, self.centerline_pts_fname)
    #     # mps_from_pts(points_fine, self.tmp_dir / 'centerline_spline.mps')

    #     # self.aorta_normal_fname = self.tmp_dir / 'aorta_normal.npy'
    #     # with open(self.aorta_normal_fname, 'wb') as f:
    #     #     np.save(f, normal)
        
    #     # spline_centerline_pts = self.spline_from_centerline_pts(centerline_pts)
    #     # mps_from_pts(spline_centerline_pts, self.centerline_pts_fname)
        
    #     return centerline_pts, normal

    # @staticmethod
    # def compute_orthogonal_basis_from_normal_vector(normal):
    #     v = np.array([1, 0, 0])
    #     if np.allclose(v, normal) or np.allclose(v, -normal):
    #         v = np.array([0, 1, 0])
    #     u1 = np.cross(normal, v)
    #     u1 = u1 / np.linalg.norm(u1)
    #     u2 = np.cross(normal, u1)
    #     u2 = u2 / np.linalg.norm(u2)
    #     return u1, u2
    
    # # @staticmethod
    # def extract_slice_from_normal_vector_and_point(self, image_data, normal, point):
    #     u1, u2 = self.compute_orthogonal_basis_from_normal_vector(normal)
    #     reslice = vtk.vtkImageReslice()
    #     reslice.SetResliceAxesDirectionCosines(u1.tolist(), u2.tolist(), normal.tolist())
    #     reslice.SetOutputDimensionality(2)
    #     reslice.SetResliceAxesOrigin(point.tolist())
    #     reslice.SetInputData(image_data)
    #     reslice.Update()
    #     output_slice = reslice.GetOutput()
    #     dims = output_slice.GetDimensions()
    #     flattened_image = vtk_to_numpy(output_slice.GetPointData().GetScalars())
    #     numpy_array = flattened_image.reshape(dims[2], dims[1], dims[0])
    #     spacing = output_slice.GetSpacing()# [::-1]
    #     # spacing = spacing[::-1][1:]
    #     if abs(normal[0]) > abs(normal[1]) and abs(normal[0]) > abs(normal[2]):
    #         relevant_spacing = (spacing[1], spacing[2]) # X-axis is perpendicular
    #     elif abs(normal[1]) > abs(normal[0]) and abs(normal[1]) > abs(normal[2]):
    #         relevant_spacing = (spacing[0], spacing[2]) # Y-axis is perpendicular
    #     else:
    #         relevant_spacing = (spacing[0], spacing[1]) # Z-axis is perpendicular
    #     return numpy_array[0], relevant_spacing

    # def aorta_slices_from_hps_and_centerline(self):#, centerline_pts):
    #     centerline_pts = pts_from_mps(self.centerline_pts_fname)
    #     hps = pts_from_mps(self.hps_fname)
    #     heart_roi_foc = sitk.ReadImage(self.heart_roi_foc_fname)
    #     centerline_pts_px = [heart_roi_foc.TransformPhysicalPointToContinuousIndex(pt) for pt in centerline_pts]
    #     heart_roi_foc.SetOrigin((0,0,0))
    #     centerline_pts_vtk = np.array([heart_roi_foc.TransformContinuousIndexToPhysicalPoint(pt) for pt in centerline_pts_px])
    #     # heart_seg = sitk.ReadImage(self.heart_seg_fname)
    #     # heart_seg_np = sitk.GetArrayFromImage(heart_seg)
    #     # aorta_seg_np = (heart_seg_np == 6).astype(np.uint8)
    #     # aorta_seg = sitk.GetImageFromArray(aorta_seg_np)
    #     # aorta_seg.CopyInformation(heart_seg)
    #     # sitk.WriteImage(aorta_seg, self.tmp_dir / 'aorta_seg_tmp_slices.nii.gz')
    #     # center_pt, radius = define_circle_3d(*hps[:3])
    #     # point, normal = plane_from_three_points(hps[:3])

    #     image_reader = vtk.vtkNIFTIImageReader()
    #     image_reader.SetFileName(self.heart_roi_foc_fname)
    #     image_reader.Update()
    #     image_data = image_reader.GetOutput()
    #     seg_reader = vtk.vtkNIFTIImageReader()
    #     seg_reader.SetFileName(self.aorta_seg_fname)
    #     seg_reader.Update()
    #     seg_data = seg_reader.GetOutput()
    #     # src_img.unlink()
    #     # src_seg.unlink()
        
    #     img_slices, seg_slices, spacings = [], [], []

    #     for i, pt in enumerate(centerline_pts_vtk[:-5]):
    #         normal_end = centerline_pts_vtk[i:i+5]
    #         normal_end = normal_end.mean(axis=0)
    #         normal = normal_end - pt
    #         image_slice, image_spacing = self.extract_slice_from_normal_vector_and_point(image_data, normal, pt)
    #         seg_slice, seg_spacing = self.extract_slice_from_normal_vector_and_point(seg_data, normal, pt)

    #         img_slices.append(image_slice)
    #         seg_slices.append(seg_slice)
    #         spacings.append(image_spacing)
            
    #     return img_slices, seg_slices, spacings, centerline_pts # _vtk

    # def render_heart(self, aorta_img_slices, aorta_seg_slices, axes_lengths):
    #     seg_keys = ['Myo', 'LA', 'LV', 'RA', 'RV', 'Aorta', 'PA']
    #     seg_colors = [(0.5,0.5,0.5), (0.5,0.5,0.5),(0,1,0),(0.5,0.5,0.5),(0,0,1),(1,0,0),(0.5,0.5,0.5)]
    #     seg_opacities = [0.6, 0.6, 0.3, 0.6, 0.6, 0.2, 0.6]
    #     heart_seg = sitk.ReadImage(self.heart_seg_fname)
    #     heart_seg_res = heart_seg # resample_image(heart_seg, out_spacing=(2,2,2))
    #     heart_seg_res_np = sitk.GetArrayFromImage(heart_seg_res)
    #     segs = []
    #     for i, n in enumerate(seg_keys):
    #     # i = 5
    #         seg_np = (heart_seg_res_np == i+1).astype(np.uint8)
    #         seg = sitk.GetImageFromArray(seg_np)
    #         seg.CopyInformation(heart_seg_res)
    #         # segs[n] = seg
    #         segs.append(seg)

    #     hps = pts_from_mps(self.hps_fname)
    #     ms = pts_from_mps(self.ms_fname)
    #     pts = np.concatenate([hps, ms], axis=0)

    #     centerline_pts = pts_from_mps(self.centerline_pts_fname)
    #     # with open(self.aorta_normal_fname, 'rb') as f:
    #     #     aorta_normal = np.load(f)

    #     renderer = VolumeRenderer(
    #         seg_object_list=segs,
    #         seg_object_colors=seg_colors,
    #         seg_object_opacities=seg_opacities,
    #         pred_pts_list=[pts],
    #         centerline_obj=centerline_pts,
    #         aorta_img_slices=aorta_img_slices,
    #         aorta_seg_slices=aorta_seg_slices,
    #         axes_lengths=axes_lengths
    #         # aorta_normal=aorta_normal
    #     )
    #     renderer.render()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--fname', default='/mnt/hdd/data/tavi_cts/example/1.2.840.113704.1.111.164.1425903939.24/CT/img.nii.gz')
    parser.add_argument('--tmp_dir', default='/mnt/hdd/data/tavi_cts/example')
    args = parser.parse_args()

    # for sid_dir in Path('/hdd/floto/fl-envoy/data/datasets/TAVI CTs with Contrast Agent').iterdir():
    # tp = TAVIPredictor(fname=str(sid_dir / 'CT' / 'img.nii.gz'), tmp_dir=str(sid_dir))
    tp = TAVIPredictor(fname=args.fname, tmp_dir=args.tmp_dir)
    if not (Path(args.tmp_dir) / 'Heart Seg').exists():
        tp.focus_heart()
        tp.segment_heart()
    else:
        tp.heart_roi_fname = tp.tmp_dir / 'Heart ROI' / 'heart_roi.nii.gz'
        tp.heart_seg_fname = tp.tmp_dir / 'Heart Seg' / 'heart_seg.nii.gz'
        tp.hinge_points()
        tp.membranous_septum()
        tp.calcification()
        if tp.hps_fname is not None and tp.ms_fname is not None and tp.calc_fname is not None:
            tp.set_origin_to_center_of_hinge_points()
            if os.environ.get('TAVI_PARAMS', None) is not None:
                tp.aorta_centerline()
                tp.tavi_parameters(tavi_params_fname='tavi_params.json')
    
    # root = Path('/mnt/hdd/data/tavi_cts/datasets/TAVI CTs with Contrast Agent')
    # patient = root / '1.2.840.113704.1.111.164.1425903939.24'
    # tp.tmp_dir = patient
    # tp.heart_seg_fname = patient / 'Heart Seg res new origin'
    # tp.hps_fname = patient / 'HP new origin'
    # tp.ms_fname = patient / 'MS new origin'
    # tp.calc_fname = patient / 'Calcification new origin'

    # tp.tavi_parameters()
