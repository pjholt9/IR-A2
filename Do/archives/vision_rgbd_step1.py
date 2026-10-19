"""Inspect an iPhone DepthCapture RGB-D snapshot and fit a candidate plane.
Usage: python Do/vision_rgbd_step1.py --sample datasets/barrel/01_center/raw
Requires numpy, matplotlib, Pillow. No robot motion is commanded.
"""
from pathlib import Path
import argparse, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_capture(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    params=json.loads((folder/'params.json').read_text(encoding='utf-8'))
    h,w=manifest['depth']['height'],manifest['depth']['width']
    depth=np.fromfile(folder/'depth.bin',dtype='<f4')
    conf=np.fromfile(folder/'confidence.bin',dtype=np.uint8)
    if depth.size!=h*w or conf.size!=h*w:
        raise ValueError(f'Binary shape mismatch: expected {h*w} pixels, got depth={depth.size}, confidence={conf.size}')
    depth=depth.reshape(h,w);conf=conf.reshape(h,w)
    rgb=np.asarray(Image.open(folder/'rgb.jpg').convert('RGB'))
    if rgb.shape[:2]!=(h,w):
        raise ValueError(f'RGB shape {rgb.shape[:2]} differs from depth {(h,w)}; alignment required')
    if params.get('intrinsicsCoordinateSpace')!='savedImage':
        raise ValueError('Intrinsics not in savedImage coordinate space; verify rotation before continuing')
    k=np.asarray(params['intrinsics'],dtype=float)
    # DepthCapture params.json stores the 3x3 intrinsics transposed:
    # [[fx,0,0],[0,fy,0],[cx,cy,1]].
    fx,fy,cx,cy=k[0,0],k[1,1],k[2,0],k[2,1]
    if not (0<=cx<w and 0<=cy<h and fx>0 and fy>0):
        raise ValueError('Invalid saved-image intrinsics')
    return rgb,depth,conf,(fx,fy,cx,cy),manifest


def point_cloud(depth,intrinsics,valid):
    fx,fy,cx,cy=intrinsics
    yy,xx=np.indices(depth.shape)
    z=depth[valid]
    x=(xx[valid]-cx)*z/fx
    y=(yy[valid]-cy)*z/fy
    return np.column_stack((x,y,z))


def fit_plane_ransac(points,iterations=800,threshold=0.008,seed=42):
    if len(points)<3: raise ValueError('Not enough points')
    rng=np.random.default_rng(seed)
    best=np.zeros(len(points),dtype=bool)
    for _ in range(iterations):
        sample=points[rng.choice(len(points),3,replace=False)]
        normal=np.cross(sample[1]-sample[0],sample[2]-sample[0])
        n=np.linalg.norm(normal)
        if n<1e-9: continue
        normal/=n
        mask=np.abs((points-sample[0])@normal)<threshold
        if mask.sum()>best.sum(): best=mask
    if best.sum()<3: raise ValueError('No plane found')
    center=points[best].mean(axis=0)
    _,_,vh=np.linalg.svd(points[best]-center,full_matrices=False)
    normal=vh[-1]
    if normal[2]<0:normal=-normal
    residual=np.abs((points-center)@normal)
    refined=residual<threshold
    return normal,center,refined


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--sample',required=True,help='Folder containing original five files')
    ap.add_argument('--output',default=None)
    ap.add_argument('--confidence',type=int,default=1,choices=[0,1,2])
    ap.add_argument('--threshold-mm',type=float,default=8.0)
    args=ap.parse_args()
    src=Path(args.sample).resolve()
    out=Path(args.output).resolve() if args.output else src.parent/'processed'
    out.mkdir(parents=True,exist_ok=True)
    rgb,depth,conf,K,manifest=load_capture(src)
    valid=np.isfinite(depth)&(depth>0)&(conf>=args.confidence)
    yy,_=np.indices(depth.shape)
    # Lower half is a heuristic for the build plate; it may still include other surfaces.
    region=valid&(yy>=int(depth.shape[0]*0.60))
    pts=point_cloud(depth,K,region)
    normal,center,inliers=fit_plane_ransac(pts,threshold=args.threshold_mm/1000)
    plane_mask=np.zeros(depth.shape,dtype=bool)
    plane_mask[region]=inliers
    fig,ax=plt.subplots(2,2,figsize=(10,10))
    ax[0,0].imshow(rgb);ax[0,0].set_title('RGB (original)')
    im=ax[0,1].imshow(np.where(valid,depth,np.nan),cmap='viridis');ax[0,1].set_title('LiDAR depth (m)');fig.colorbar(im,ax=ax[0,1],fraction=.046)
    im=ax[1,0].imshow(conf,vmin=0,vmax=2,cmap='plasma');ax[1,0].set_title('Confidence 0=low, 2=high');fig.colorbar(im,ax=ax[1,0],fraction=.046)
    ax[1,1].imshow(rgb);ax[1,1].imshow(np.ma.masked_where(~plane_mask,plane_mask),cmap='autumn',alpha=.65,vmin=0,vmax=1)
    ax[1,1].set_title('RANSAC plane candidate (lower 40% ROI)')
    for a in ax.flat:a.axis('off')
    fig.tight_layout();fig.savefig(out/'rgbd_diagnostics.png',dpi=170);plt.close(fig)
    np.save(out/'points_lower_roi_camera.npy',pts)
    result={'source':str(src),'image_shape_hw':list(depth.shape),'camera_intrinsics_saved_image':dict(zip(('fx','fy','cx','cy'),map(float,K))),'depth_unit':'meters','confidence_min':args.confidence,'valid_pixels':int(valid.sum()),'total_pixels':int(valid.size),'candidate_plane_normal_camera':normal.tolist(),'candidate_plane_point_camera_m':center.tolist(),'candidate_plane_inliers':int(inliers.sum()),'candidate_plane_points_tested':int(len(pts)),'note':'RANSAC finds a plane candidate, not verified as the build plate. Camera frame is optical: x right, y down, z forward. No robot/world registration or grasp pose estimated.'}
    (out/'diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));print('Saved:',out/'rgbd_diagnostics.png')

if __name__=='__main__':main()
