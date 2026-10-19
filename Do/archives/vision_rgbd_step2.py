"""Step 2: single-frame RGB-D barrel segmentation and 3D diagnostics.

No robot commands. Estimates a provisional barrel mask, its visible 3D surface,
and a plane-relative *surface* location; does NOT claim a calibrated grasp pose.

From IR-A2 root:
  .\\.venv\\Scripts\\python.exe Do\\vision_rgbd_step2.py --sample Do\\datasets\\barrel\\01_center\\raw
Dependencies: numpy, matplotlib, pillow, opencv-python.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from vision_rgbd_step1 import load_capture, fit_plane_ransac, point_cloud


def run(sample, output=None, confidence=1, plane_threshold_mm=8.0):
    sample = Path(sample).resolve()
    output = Path(output).resolve() if output else sample.parent / 'processed'
    output.mkdir(parents=True, exist_ok=True)
    rgb, depth, conf, K, manifest = load_capture(sample)
    h, w = depth.shape
    yy, xx = np.indices((h, w))
    valid = np.isfinite(depth) & (depth > 0) & (conf >= confidence)
    plane_roi = valid & (yy >= int(h * 0.60))
    plane_points = point_cloud(depth, K, plane_roi)
    normal, origin, plane_inliers = fit_plane_ransac(plane_points, threshold=plane_threshold_mm / 1000)
    xyz = np.stack(((xx - K[2]) * depth / K[0], (yy - K[3]) * depth / K[1], depth), axis=-1)
    signed = np.einsum('ijk,k->ij', xyz - origin, normal)

    # Determine plane-normal direction from a central point above the plate.
    # In this dataset the object's points have negative signed distances.
    central = valid & (xx > int(w*.40)) & (xx < int(w*.60)) & (yy > int(h*.18)) & (yy < int(h*.55))
    sign = -1.0 if np.median(signed[central]) < 0 else 1.0
    height = sign * signed
    up = sign * normal

    # GrabCut is seeded using a coarse object ROI only, NOT ground-truth pose.
    # The bounding rectangle is a sample-specific initialization and should
    # be validated for each of the five positions before generalization.
    rect = (int(w*.335), int(h*.055), int(w*.285), int(h*.73))
    mask = np.zeros((h, w), np.uint8)
    bg_model = np.zeros((1,65), np.float64)
    fg_model = np.zeros((1,65), np.float64)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.grabCut(bgr, mask, rect, bg_model, fg_model, 8, cv2.GC_INIT_WITH_RECT)
    grab = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)

    # Keep the connected component overlapping the central barrel region.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(grab.astype('uint8'), 8)
    scores = []
    for i in range(1,count):
        x,y,bw,bh,area = stats[i]
        overlap = np.sum((labels == i) & (xx > w*.41) & (xx < w*.59) & (yy > h*.15) & (yy < h*.58))
        scores.append((overlap,area,i))
    chosen = max(scores)[2] if scores else None
    rgb_mask = (labels == chosen) if chosen is not None else np.zeros((h,w),bool)
    # Depth height gate removes most of the build plate and near-plane pixels.
    barrel = rgb_mask & valid & (height > .012) & (height < .32)
    # Restrict 3D estimate to high-confidence visible barrel pixels.
    barrel_pts = xyz[barrel]
    if len(barrel_pts) < 60:
        raise RuntimeError(f'Only {len(barrel_pts)} segmented 3D points; review mask and GrabCut ROI')

    # Estimate projected footprint center using visible lower body points.
    # This is a surface proxy, not a measured center of the full cylinder.
    lower = barrel & (height >= .025) & (height <= .085)
    proxy = np.median(xyz[lower],axis=0) if lower.sum() >= 20 else np.median(barrel_pts,axis=0)
    proxy_height = float(np.dot(proxy-origin,up))
    proxy_on_plane = proxy - proxy_height*up
    grasp_surface_proxy = proxy_on_plane + .160*up

    # Plane RANSAC mask for verification.
    plane_mask = np.zeros((h,w),bool)
    plane_mask[plane_roi] = plane_inliers
    fig,axs=plt.subplots(2,3,figsize=(15,10))
    axs[0,0].imshow(rgb);axs[0,0].set_title('RGB original')
    im=axs[0,1].imshow(np.where(valid,depth,np.nan),cmap='viridis');axs[0,1].set_title('Depth (m)');fig.colorbar(im,ax=axs[0,1],fraction=.045)
    im=axs[0,2].imshow(np.where(valid,height,np.nan),cmap='coolwarm',vmin=-.02,vmax=.24);axs[0,2].set_title('Height above candidate plate (m)');fig.colorbar(im,ax=axs[0,2],fraction=.045)
    axs[1,0].imshow(rgb);axs[1,0].imshow(np.ma.masked_where(~rgb_mask,rgb_mask),cmap='spring',alpha=.5);axs[1,0].set_title('GrabCut RGB segmentation')
    axs[1,1].imshow(rgb);axs[1,1].imshow(np.ma.masked_where(~barrel,barrel),cmap='autumn',alpha=.65);axs[1,1].set_title('RGB + depth barrel candidate')
    axs[1,2].imshow(rgb);axs[1,2].imshow(np.ma.masked_where(~plane_mask,plane_mask),cmap='summer',alpha=.55);axs[1,2].set_title('RANSAC plate candidate')
    for ax in axs.flat:ax.axis('off')
    fig.tight_layout();fig.savefig(output/'barrel_step2_diagnostics.png',dpi=160);plt.close(fig)

    np.save(output/'barrel_points_camera.npy',barrel_pts)
    np.save(output/'barrel_mask.npy',barrel)
    result={
        'status':'PROVISIONAL_SEGMENTATION_NOT_ROBOT_READY',
        'sample':str(sample),
        'image_shape_hw':[h,w],
        'candidate_plate_normal_toward_barrel_camera':up.tolist(),
        'candidate_plate_point_camera_m':origin.tolist(),
        'grabcut_rectangle_xywh':list(rect),
        'rgb_mask_pixels':int(rgb_mask.sum()),
        'barrel_depth_mask_pixels':int(barrel.sum()),
        'barrel_height_percentiles_m':np.percentile(height[barrel],[5,50,95]).tolist(),
        'barrel_visible_surface_median_camera_m':np.median(barrel_pts,axis=0).tolist(),
        'lower_barrel_visible_surface_proxy_camera_m':proxy.tolist(),
        'lower_surface_proxy_projected_to_plane_camera_m':proxy_on_plane.tolist(),
        'grasp_height_160mm_surface_proxy_camera_m':grasp_surface_proxy.tolist(),
        'warnings':[
            'GrabCut rectangle is tuned for center sample; not validated on other locations.',
            'The 160 mm proxy uses a visible surface point, NOT the barrel center axis.',
            'No camera-to-Swift registration, true 3D cylinder fit, or safe grasp pose.',
            'Do not feed these coordinates directly to RobotTasks.'
        ]
    }
    (output/'barrel_step2_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    print('Saved:',output/'barrel_step2_diagnostics.png')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--sample',required=True)
    ap.add_argument('--output')
    ap.add_argument('--confidence',type=int,default=1,choices=[0,1,2])
    ap.add_argument('--threshold-mm',type=float,default=8.)
    a=ap.parse_args()
    run(a.sample,a.output,a.confidence,a.threshold_mm)
