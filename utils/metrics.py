import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import binary_erosion
import cv2

try:
    from medpy.metric.binary import jc as medpy_jc, dc as medpy_dc, hd as medpy_hd, hd95 as medpy_hd95
    from medpy.metric.binary import recall as medpy_recall, specificity as medpy_specificity, precision as medpy_precision
    MEDPY_AVAILABLE = True
    print("✅ medpy，HD")
except ImportError:
    MEDPY_AVAILABLE = False
    print("⚠️  medpy，HD")

def jc(result, reference):
    """Jaccard coefficient (IoU)"""
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    intersection = np.count_nonzero(result & reference)
    union = np.count_nonzero(result | reference)

    if union == 0:
        return 1.0
    return intersection / union

def dc(result, reference):
    """Dice coefficient"""
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    intersection = np.count_nonzero(result & reference)
    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    if size_i1 + size_i2 == 0:
        return 1.0
    return 2. * intersection / (size_i1 + size_i2)

def get_surface_points_fallback(mask):
    """
    ：
    OpenCV
    """
    from scipy.ndimage import binary_erosion
    
    if mask.sum() == 0:
        return np.array([]).reshape(0, 2)
    
    if mask.ndim > 2:
        mask = mask.squeeze()
    mask_bool = mask.astype(np.bool_)
    
    eroded = binary_erosion(mask_bool)
    boundary = mask_bool & ~eroded
    
    boundary_coords = np.argwhere(boundary)
    
    if len(boundary_coords) == 0:
        return np.argwhere(mask_bool)
    
    return boundary_coords

def get_surface_points(mask):
    """
    mask/
    OpenCV，
    """
    if mask.sum() == 0:
        return np.array([]).reshape(0, 2)
    
    if mask.ndim > 2:
        mask = mask.squeeze()
    
    try:
        mask_clean = mask.astype(np.bool_).astype(np.uint8) * 255
        
        if mask_clean.ndim > 2:
            mask_clean = mask_clean.squeeze()
        mask_clean = np.ascontiguousarray(mask_clean, dtype=np.uint8)
        
        if len(mask_clean.shape) != 2:
            raise ValueError(f"Mask shape is {mask_clean.shape}, expected 2D")
        
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if contours:
            surface_points = []
            for contour in contours:
                points = contour.reshape(-1, 2)
                surface_points.append(points)
            
            if surface_points:
                return np.vstack(surface_points)
        
        return get_surface_points_fallback(mask)
        
    except (cv2.error, ValueError, Exception) as e:
        print(f"OpenCV，: {str(e)[:100]}...")
        return get_surface_points_fallback(mask)

def compute_surface_distances(surface1, surface2):
    """
    """
    if len(surface1) == 0 or len(surface2) == 0:
        return np.array([])
    
    distances = []
    for point in surface1:
        dists = np.sqrt(np.sum((surface2 - point) ** 2, axis=1))
        distances.append(np.min(dists))
    
    return np.array(distances)

def hd(result, reference, voxelspacing=None, connectivity=1):
    """
    Hausdorff Distance
    HD = max(max(d(s1->s2)), max(d(s2->s1)))
    """
    try:
        result = result.astype(np.bool_)
        reference = reference.astype(np.bool_)
        
        if np.count_nonzero(result) == 0 or np.count_nonzero(reference) == 0:
            if np.count_nonzero(result) == 0 and np.count_nonzero(reference) == 0:
                return 0.0
            else:
                return float('inf')
        
        surface_result = get_surface_points(result)
        surface_reference = get_surface_points(reference)
        
        if len(surface_result) == 0 or len(surface_reference) == 0:
            return float('inf')
        
        distances_1_to_2 = compute_surface_distances(surface_result, surface_reference)
        distances_2_to_1 = compute_surface_distances(surface_reference, surface_result)
        
        if len(distances_1_to_2) == 0 or len(distances_2_to_1) == 0:
            return float('inf')
        
        # Hausdorff Distance = max of both directions
        hd_1_to_2 = np.max(distances_1_to_2)
        hd_2_to_1 = np.max(distances_2_to_1)
        
        return max(hd_1_to_2, hd_2_to_1)
        
    except Exception as e:
        print(f"HD calculation error: {e}")
        return float('inf')

def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile Hausdorff Distance
    HD95 = 95th percentile of all surface distances
    """
    try:
        result = result.astype(np.bool_)
        reference = reference.astype(np.bool_)
        
        if np.count_nonzero(result) == 0 or np.count_nonzero(reference) == 0:
            if np.count_nonzero(result) == 0 and np.count_nonzero(reference) == 0:
                return 0.0
            else:
                return float('inf')
        
        surface_result = get_surface_points(result)
        surface_reference = get_surface_points(reference)
        
        if len(surface_result) == 0 or len(surface_reference) == 0:
            return float('inf')
        
        distances_1_to_2 = compute_surface_distances(surface_result, surface_reference)
        distances_2_to_1 = compute_surface_distances(surface_reference, surface_result)
        
        if len(distances_1_to_2) == 0 or len(distances_2_to_1) == 0:
            return float('inf')
        
        all_distances = np.concatenate([distances_1_to_2, distances_2_to_1])
        
        hd95_value = np.percentile(all_distances, 95)
        
        return hd95_value
        
    except Exception as e:
        print(f"HD95 calculation error: {e}")
        return float('inf')

def recall(result, reference):
    """Sensitivity/Recall"""
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)

    if tp + fn == 0:
        return 1.0
    return tp / (tp + fn)

def specificity(result, reference):
    """Specificity"""
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)

    if tn + fp == 0:
        return 1.0
    return tn / (tn + fp)

def precision(result, reference):
    """Precision"""
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    tp = np.count_nonzero(result & reference)
    fp = np.count_nonzero(result & ~reference)

    if tp + fp == 0:
        return 1.0
    return tp / (tp + fp)



def iou_score(output, target):
    """
    IoU、DiceHD95，Seg_UKAN
    ：(iou, dice, hd95)
    """
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)
    
    try:
        if MEDPY_AVAILABLE:
            if output_.ndim > 2:
                output_2d = output_.squeeze()
                target_2d = target_.squeeze()
            else:
                output_2d = output_
                target_2d = target_
            hd95_ = medpy_hd95(output_2d, target_2d)
        else:
            if output_.ndim > 2:
                output_2d = output_.squeeze()
                target_2d = target_.squeeze()
            else:
                output_2d = output_
                target_2d = target_
            hd95_ = hd95(output_2d, target_2d)
    except:
        hd95_ = 0
    
    return iou, dice, hd95_


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)

def indicators(output, target):
    """
    ，medpy（Seg_UKAN），
    """
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    if MEDPY_AVAILABLE:
        try:
            if output_.ndim > 2:
                output_2d = output_.squeeze()
                target_2d = target_.squeeze()
            else:
                output_2d = output_
                target_2d = target_
            
            iou_ = medpy_jc(output_2d, target_2d)
            dice_ = medpy_dc(output_2d, target_2d)
            
            try:
                hd_ = medpy_hd(output_2d, target_2d)
            except:
                hd_ = 0
                
            try:
                hd95_ = medpy_hd95(output_2d, target_2d)
            except:
                hd95_ = 0
            
            recall_ = medpy_recall(output_2d, target_2d)
            specificity_ = medpy_specificity(output_2d, target_2d)
            precision_ = medpy_precision(output_2d, target_2d)
            
            return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_
            
        except Exception as e:
            print(f"medpy，: {str(e)[:50]}...")
    
    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)

    try:
        if output_.ndim > 2:
            output_2d = output_.squeeze()
            target_2d = target_.squeeze()
        else:
            output_2d = output_
            target_2d = target_
            
        hd_ = hd(output_2d, target_2d)
        hd95_ = hd95(output_2d, target_2d)
        
        if np.isinf(hd_):
            img_shape = output_2d.shape
            max_distance = np.sqrt(img_shape[0]**2 + img_shape[1]**2)
            hd_ = max_distance
            
        if np.isinf(hd95_):
            img_shape = output_2d.shape
            max_distance = np.sqrt(img_shape[0]**2 + img_shape[1]**2)
            hd95_ = max_distance
            
    except Exception as e:
        print(f"HD: {e}")
        img_shape = output_.shape[-2:] if output_.ndim > 2 else output_.shape
        max_distance = np.sqrt(img_shape[0]**2 + img_shape[1]**2)
        hd_ = max_distance
        hd95_ = max_distance

    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)

    return iou_, dice_, hd_, hd95_, recall_, specificity_, precision_
