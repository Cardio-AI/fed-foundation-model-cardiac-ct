import raster_geometry as rg
import cc3d
from scipy import ndimage
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import SimpleITK as sitk
import itk
from bs4 import BeautifulSoup
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn as get_tp_fp_fn_tn_nnunet
from src.data.utils import resample_image

def get_tp_fp_fn_tn(pred, target):
    tp = ((pred == 1) * (target == 1)).sum()
    fp = ((pred == 1) * (target == 0)).sum()
    fn = ((pred == 0) * (target == 1)).sum()
    tn = ((pred == 0) * (target == 0)).sum()
    return tp, fp, fn, tn

def itk_to_sitk(itk_image):
    np_array = itk.GetArrayFromImage(itk_image)
    sitk_image = sitk.GetImageFromArray(np_array)
    spacing = itk_image.GetSpacing()
    sitk_image.SetSpacing([float(spacing[i]) for i in range(itk_image.GetImageDimension())])
    origin = itk_image.GetOrigin()
    sitk_image.SetOrigin([float(origin[i]) for i in range(itk_image.GetImageDimension())])
    direction = itk_image.GetDirection()
    sitk_direction = [direction.GetVnlMatrix().get(i, j) for i in range(itk_image.GetImageDimension()) for j in range(itk_image.GetImageDimension())]
    sitk_image.SetDirection(sitk_direction)
    return sitk_image

def pts_from_pred(pred, ref_sitk):
    if type(pred) == torch.Tensor:
        pred = pred.cpu().detach().numpy()
    largest_vols = find_largest_volume(pred)
    pts_idx_pred = find_center_of_mass(largest_vols).tolist()
    # if type(ref_sitk) != sitk.Image:
    pts_pred = np.array([ref_sitk.TransformContinuousIndexToPhysicalPoint(pt_idx[::-1]) for pt_idx in pts_idx_pred])
    # else:
    #     pts_idx_pred_itk = [itk.ContinuousIndex[itk.F, 3](i) for i in pts_idx_pred]
    #     pts_pred = np.array([ref_sitk.TransformContinuousIndexToPhysicalPoint(pt_idx) for pt_idx in pts_idx_pred_itk])
    return pts_idx_pred, pts_pred

def pts_from_pred_onehot(pred_onehot, ref_sitk):
    if type(pred_onehot) == torch.Tensor:
        pred_onehot = pred_onehot.cpu().detach().numpy()
    pts_pred, pts_idx_pred = [], []
    for i in range(pred_onehot.shape[0]):
        p = pred_onehot[i]
        if len(np.unique(p)) < 2:
            continue
        largest_vols = find_largest_volume(p)
        i_pts_idx_pred = find_center_of_mass(largest_vols).tolist()
        i_pts_pred = np.array([ref_sitk.TransformContinuousIndexToPhysicalPoint(pt_idx[::-1]) for pt_idx in i_pts_idx_pred])
        pts_pred.append(i_pts_pred[0])
        pts_idx_pred.append(i_pts_idx_pred[0])
    return np.array(pts_idx_pred), np.array(pts_pred)

def pred_from_onehot(pred_onehot):
    bg_pred = (~pred_onehot.sum(dim=1, keepdim=True).bool()).float()
    pred_onehot = torch.cat([bg_pred, pred_onehot], dim=1)
    pred = torch.argmax(pred_onehot, dim=1).float()#[0]
    return pred

def hard_dice_score(pred_onehot, target):
    tp, fp, fn, _ = get_tp_fp_fn_tn_nnunet(pred_onehot, target, axes=(0,2,3,4))
    tp = tp.detach().cpu().numpy()
    fp = fp.detach().cpu().numpy()
    fn = fn.detach().cpu().numpy()
    dice = 2 * tp / (2 * tp + fp + fn + 1)
    return dice

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def plane_from_three_points(points):
    p0, p1, p2 = points
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    ux, uy, uz = u = [x1-x0, y1-y0, z1-z0]
    vx, vy, vz = v = [x2-x0, y2-y0, z2-z0]

    u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]

    point  = np.array(p0)
    normal = np.array(u_cross_v)
    normal /= np.linalg.norm(normal)
    return point, normal

def compute_plane_idx_from_three_points(img_np, points):
    point, normal = plane_from_three_points(points)

    d = -point.dot(normal)
    xx, yy = np.meshgrid(range(img_np.shape[0]), range(img_np.shape[1]))
    zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    zz = np.round(zz).astype(int)
    
    return xx, yy, zz

def point_distance_from_plane(p_plane, n_plane, p):
    # https://stackoverflow.com/questions/55189333/how-to-get-distance-from-point-to-plane-in-3d
    # https://mathinsight.org/distance_point_plane
    n_plane = n_plane / np.linalg.norm(n_plane)
    distance = np.abs(np.dot(p - p_plane, n_plane))
    return distance

def rotate_to_aortic_annulus_plane_view(hinge_points, img_np, img_sitk, slices_down=-10, slices_up=80):
    hinge_points = np.array([img_sitk.TransformPhysicalPointToIndex(hp.tolist()) for hp in hinge_points])
    xx, yy, zz = compute_plane_idx_from_three_points(img_np, hinge_points)
    img_np_rot = np.concatenate([img_np[xx,yy,np.clip(zz+i, 0, img_np.shape[-1]-1)][...,None] for i in range(slices_down, slices_up,1)], axis=-1)
    return img_np_rot

def define_circle(p1, p2, p3):
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
    
    if abs(det) < 1.0e-6:
        return (None, np.inf)
    
    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
    
    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)

def define_circle_3d(A, B, C):
    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(B - A)
    s = (a + b + c) / 2
    R = a*b*c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
    b1 = a*a * (b*b + c*c - a*a)
    b2 = b*b * (a*a + c*c - b*b)
    b3 = c*c * (a*a + b*b - c*c)
    P = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
    P /= b1 + b2 + b3
    return P, R

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

def find_center_of_mass(seg: np.ndarray) -> np.array:
    #seg_labeled = ndimage.label(seg)[0]
    centers = ndimage.center_of_mass(seg, seg, np.unique(seg)[1:])
    return np.array(centers)

def compute_distance(hp_true, hp_pred):
    eucl_dist = np.linalg.norm(hp_true - hp_pred, axis=1)
    return eucl_dist

def focus_valve_region_from_pts(img, pts_idx=None, pts=None, img_size=None, step_mm=None, step_px=None):
    pts = pts.reshape(-1,3)
    mins, maxs = np.min(pts, axis=0), np.max(pts, axis=0)
    mins -= step_mm
    maxs += step_mm
    min_x, min_y, min_z = img.TransformPhysicalPointToIndex(mins)
    max_x, max_y, max_z = img.TransformPhysicalPointToIndex(maxs)
    x_dim, y_dim, z_dim = img.GetSize()
    img_focused = img[
        max(0,min_x):min(x_dim,max_x),
        max(0,min_y):min(y_dim,max_y),
        max(0,min_z):min(z_dim,max_z)
    ]
    if img_size is not None:
        img_focused = resample_image(img_focused, new_size=img_size)
    return img_focused

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

def pts_from_mps(path):
    with open(path, 'r') as f:
        pts = f.read()
    
    soup = BeautifulSoup(pts, "xml")
    points = soup.find_all('point')
    extract_coordinates = lambda p: [
        float(p.findChildren('x')[0].text), float(p.findChildren('y')[0].text), float(p.findChildren('z')[0].text)
    ]
    coordinates = [extract_coordinates(p) for p in points]
    # description = [[f'x{i}', f'y{i}', f'z{i}'] for i in range(len(coordinates))]
    coordinates = [float(xx) for x in coordinates for xx in x]
    # description = [str(xx) for x in description for xx in x]

    return np.array(coordinates).reshape(-1, 3)

def onehot(t, n_cls):
    b, _, h, w, d = t.shape
    t_onehot = torch.zeros((b, n_cls, h, w, d)).to(t.device)
    t_onehot.scatter_(1, t.long(), 1)
    return t_onehot

def uncertainty(p_hat, var='sum'):
    p_mean = torch.mean(p_hat, dim=0)
    ale = torch.mean(p_hat*(1-p_hat), dim=0)
    epi = torch.mean(p_hat**2, dim=0) - p_mean**2
    if var == 'sum':
        ale = torch.sum(ale, dim=0)
        epi = torch.sum(epi, dim=0)
    elif var == 'top':
        ale = ale[torch.argmax(p_mean)]
        epi = epi[torch.argmax(p_mean)]
    uncert = ale + epi
    return p_mean, uncert, ale, epi

class MetricsTicker:
    heart_keys = ['Myo', 'LA', 'LV', 'RA', 'RV', 'Aorta', 'PA']
    hps_keys = ['RCC', 'LCC', 'ACC', 'RCA', 'LCA']
    ms_keys = ['MS1', 'MS2']
    calc_keys = ['Calc']
    pm_keys = ['tp', 'fp', 'fn', 'tn']

    def __init__(self):
        self.heart_dices = {k: [] for k in self.heart_keys}
        self.hps_dices = {k: [] for k in self.hps_keys}
        self.ms_dices = {k: [] for k in self.ms_keys}
        self.calc_dices = {k: [] for k in self.calc_keys}
        self.hps_distances = {k: [] for k in self.hps_keys}
        self.ms_distances = {k: [] for k in self.ms_keys}
        self.pm_cls_metrics = {k: [] for k in self.pm_keys}
    
    def append(self, key, values, distances=None):
        if key == 'heart':
            for k, v in zip(self.heart_keys, values):
                self.heart_dices[k].append(v)
        elif key == 'hps':
            for k, v, d in zip(self.hps_keys, values, distances):
                self.hps_dices[k].append(v)
                self.hps_distances[k].append(d)
        elif key == 'ms':
            for k, v, d in zip(self.ms_keys, values, distances):
                self.ms_dices[k].append(v)
                self.ms_distances[k].append(d)
        elif key == 'calc':
            for k, v in zip(self.calc_keys, values):
                self.calc_dices[k].append(v)
        elif key == 'pm':
            for k, v in zip(self.pm_keys, values):
                self.pm_cls_metrics[k].append(v)
    
    def append_summary(self, summary):
        if 'heart_dice' in summary:
            for k, v in summary['heart_dice'].items():
                self.heart_dices[k].append(v)
        if 'hps_dice' in summary:
            for k, v in summary['hps_dice'].items():
                self.hps_dices[k].append(v)
        if 'hps_distance' in summary:
            for k, v in summary['hps_distance'].items():
                self.hps_distances[k].append(v)
        if 'ms_dice' in summary:
            for k, v in summary['ms_dice'].items():
                self.ms_dices[k].append(v)
        if 'ms_distance' in summary:
            for k, v in summary['ms_distance'].items():
                self.ms_distances[k].append(v)
        if 'calc_dice' in summary:
            for k, v in summary['calc_dice'].items():
                self.calc_dices[k].append(v)
        if 'pm_cls_metrics' in summary:
            for k, v in summary['pm_cls_metrics'].items():
                self.pm_cls_metrics[k].append(v)
    
    def summarize(self):
        metrics = {}
        if len(list(self.heart_dices.values())[0]):
            heart_dices = {k: np.mean(v) for k, v in self.heart_dices.items()}
            metrics['heart_dice'] = heart_dices
        if len(list(self.hps_dices.values())[0]):
            hps_dices = {k: np.mean(v) for k, v in self.hps_dices.items()}
            metrics['hps_dice'] = hps_dices
            hps_distances = {k: np.nanmean(np.array(v)) for k, v in self.hps_distances.items()}
            metrics['hps_distance'] = hps_distances
        if len(list(self.ms_dices.values())[0]):
            ms_dices = {k: np.mean(v) for k, v in self.ms_dices.items()}
            metrics['ms_dice'] = ms_dices
            ms_distances = {k: np.nanmean(np.array(v)) for k, v in self.ms_distances.items()}
            metrics['ms_distance'] = ms_distances
        if len(list(self.calc_dices.values())[0]):
            calc_dices = {k: np.mean(v) for k, v in self.calc_dices.items()}
            metrics['calc_dice'] = calc_dices
        if len(self.pm_cls_metrics):
            pm_cls_metrics = {k: np.sum(v) for k, v in self.pm_cls_metrics.items()}
            metrics['pm_cls_metrics'] = pm_cls_metrics
        return metrics

    def plot(self, axs, label_prefix=''):
        total_pos = self.pm_cls_metrics['tp'][-1] + self.pm_cls_metrics['fn'][-1]
        total_neg = self.pm_cls_metrics['fp'][-1] + self.pm_cls_metrics['tn'][-1]
        axs[0,0].plot(np.array(self.pm_cls_metrics['tp']) / total_pos, label=label_prefix)
        axs[0,1].plot(np.array(self.pm_cls_metrics['fp']) / total_neg, label=label_prefix)
        axs[1,0].plot(np.array(self.pm_cls_metrics['fn']) / total_pos, label=label_prefix)
        axs[1,1].plot(np.array(self.pm_cls_metrics['tn']) / total_neg, label=label_prefix)
        cp = sns.color_palette()
        for i, (k, v) in enumerate(self.heart_dices.items()):
            axs[0,2].plot(v, label=k if label_prefix == 'train' else None, linestyle='-' if label_prefix == 'train' else '--', color=cp[i])
        for i, (k, v) in enumerate(self.calc_dices.items()):
            axs[1,2].plot(v, label=k if label_prefix == 'train' else None, linestyle='-' if label_prefix == 'train' else '--', color=cp[i])
        for i, (k, v) in enumerate(self.hps_dices.items()):
            axs[0,3].plot(v, label=k if label_prefix == 'train' else None, linestyle='-' if label_prefix == 'train' else '--', color=cp[i])
        for i, (k, v) in enumerate(self.ms_dices.items()):
            axs[1,3].plot(v, label=k if label_prefix == 'train' else None, linestyle='-' if label_prefix == 'train' else '--', color=cp[i])
        for i, (k, v) in enumerate(self.hps_distances.items()):
            axs[0,4].plot(v, label=k if label_prefix == 'train' else None, linestyle='-' if label_prefix == 'train' else '--', color=cp[i])
        for i, (k, v) in enumerate(self.ms_distances.items()):
            axs[1,4].plot(v, label=k if label_prefix == 'train' else None, linestyle='-' if label_prefix == 'train' else '--', color=cp[i])
        for ax in axs[:,:-1].flatten():
            ax.set_ylim([-0.1,1.1])

    def save(self, path):
        metrics = {}
        if len(list(self.heart_dices.values())[0]):
            metrics['heart_dice'] = self.heart_dices
        if len(list(self.hps_dices.values())[0]):
            metrics['hps_dice'] = self.hps_dices
            metrics['hps_distance'] = self.hps_distances
        if len(list(self.ms_dices.values())[0]):
            metrics['ms_dice'] = self.ms_dices
            metrics['ms_distance'] = self.ms_distances
        if len(list(self.calc_dices.values())[0]):
            metrics['calc_dice'] = self.calc_dices
        if len(self.pm_cls_metrics):
            metrics['pm_cls_metrics'] = self.pm_cls_metrics
        torch.save(metrics, path)

    def load(self, path):
        metrics = torch.load(path)
        if 'heart_dice' in metrics:
            self.heart_dices = metrics['heart_dice']
        if 'hps_dice' in metrics:
            self.hps_dices = metrics['hps_dice']
        if 'hps_distance' in metrics:
            self.hps_distances = metrics['hps_distance']
        if 'ms_dice' in metrics:
            self.ms_dices = metrics['ms_dice']
        if 'ms_distance' in metrics:
            self.ms_distances = metrics['ms_distance']
        if 'calc_dice' in metrics:
            self.calc_dices = metrics['calc_dice']
        if 'pm_cls_metrics' in metrics:
            self.pm_cls_metrics = metrics['pm_cls_metrics']


##############################
            
import numpy as np
from sklearn.metrics import roc_auc_score
import torch

def get_tp_fp_fn_tn(pred, gt):
    tp = ((pred==1) * (gt==1)).sum()
    fp = ((pred==1) * (gt==0)).sum()
    fn = ((pred==0) * (gt==1)).sum()
    tn = ((pred==0) * (gt==0)).sum()
    return tp, fp, fn, tn

class ECGClassificationMetrics:
    def __init__(self):
        self.accuracy = []
        self.sensitivity = []
        self.specificity = []
        self.tp = []
        self.fp = []
        self.fn = []
        self.tn = []
        self.roc_auc = []
        self.loss = []

    def compute_per_class_tp_fp_fn_tn(self, pred, gt):
        tp_fp_fn_tn = [get_tp_fp_fn_tn(pred[:,i], gt[:,i]) for i in range(gt.shape[1])]
        for i, m in enumerate([self.tp, self.fp, self.fn, self.tn]):
            m.append([x[i] for x in tp_fp_fn_tn])
    
    def compute_per_class_roc_auc_score(self, sigmoids, gt):
        try:
            roc_aucs = [roc_auc_score(gt[:,i], sigmoids[:,i]) for i in range(gt.shape[1])]
            self.roc_auc.append(roc_aucs)
        except ValueError:
            # Only one class present -> roc_auc not defined in that case
            pass
    
    def summarize(self):
        n_classes = len(self.tp[0])
        tp = [np.sum([x[i] for x in self.tp]) for i in range(n_classes)]
        fp = [np.sum([x[i] for x in self.fp]) for i in range(n_classes)]
        fn = [np.sum([x[i] for x in self.fn]) for i in range(n_classes)]
        tn = [np.sum([x[i] for x in self.tn]) for i in range(n_classes)]
        roc_auc = [np.mean([x[i] for x in self.roc_auc]) for i in range(n_classes)]
        acc = [(tp[i]+tn[i]) / (tp[i]+fp[i]+fn[i]+tn[i]+1e-6) for i in range(n_classes)]
        sensitivity = [tp[i] / (tp[i]+fn[i]+1e-6) for i in range(n_classes)]
        specificity = [tn[i] / (fp[i]+tn[i]+1e-6) for i in range(n_classes)]
        loss = np.mean(self.loss)
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, roc_auc=roc_auc, accuracy=acc, sensitivity=sensitivity, specificity=specificity, loss=loss)
    
    def append(self, tp, fp, fn, tn, roc_auc, accuracy, sensitivity, specificity, loss):
        self.tp.append(tp)
        self.fp.append(fp)
        self.fn.append(fn)
        self.tn.append(tn)
        self.roc_auc.append(roc_auc)
        self.accuracy.append(accuracy)
        self.sensitivity.append(sensitivity)
        self.specificity.append(specificity) 
        self.loss.append(loss)

    def plot(self, axs, label_prefix=''):
        axs[0].plot(self.loss, label=f'{label_prefix}')
        n_classes = len(self.tp[0])
        for i in range(n_classes):
            total_pos = self.tp[-1][i] + self.fn[-1][i]
            total_neg = self.fp[-1][i] + self.tn[-1][i]
            axs[1].plot([x[i] for x in self.accuracy], label=f'{label_prefix}_cls{i}')
            axs[2].plot([x[i] for x in self.sensitivity], label=f'{label_prefix}_cls{i}')
            axs[3].plot([x[i] for x in self.specificity], label=f'{label_prefix}_cls{i}')
            axs[4].plot([x[i] / total_pos for x in self.tp], label=f'{label_prefix}_cls{i}')
            axs[5].plot([x[i] / total_neg for x in self.fp], label=f'{label_prefix}_cls{i}')
            axs[6].plot([x[i] / total_pos for x in self.fn], label=f'{label_prefix}_cls{i}')
            axs[7].plot([x[i] / total_neg for x in self.tn], label=f'{label_prefix}_cls{i}')
            # axs[8].plot([x[i] for x in self.roc_auc], label=f'{label_prefix}_cls{i}')
        axs[0].set_title('Loss')
        axs[1].set_title('Accuracy')
        axs[2].set_title('Sensitivity')
        axs[3].set_title('Specificity')
        axs[4].set_title('TP')
        axs[5].set_title('FP')
        axs[6].set_title('FN')
        axs[7].set_title('TN')
        # axs[8].set_title('ROC AUC')
        for ax in axs[1:]:
            ax.set_ylim([-0.02, 1.02])

    def save(self, path):
        metrics = dict(
            tp=self.tp, 
            fp=self.fp, 
            fn=self.fn, 
            tn=self.tn, 
            roc_auc=self.roc_auc, 
            accuracy=self.accuracy, 
            sensitivity=self.sensitivity, 
            specificity=self.specificity, 
            loss=self.loss
        )
        torch.save(metrics, path)
    
    def load(self, path):
        metrics = torch.load(path)
        self.tp = metrics['tp']
        self.fp = metrics['fp']
        self.fn = metrics['fn']
        self.tn = metrics['tn']
        self.accuracy = metrics['accuracy']
        self.specificity = metrics['specificity']
        self.sensitivity = metrics['sensitivity']
        self.roc_auc = metrics['roc_auc']
        self.loss = metrics['loss']
