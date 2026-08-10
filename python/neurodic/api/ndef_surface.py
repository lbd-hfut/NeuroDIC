"""Thin assembly/export layer for compiled NDeF reference-surface pretraining."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from ..config import load_config
from ..ndef_paths import camera_name_from_label, ndef_run_roots
from ..models import _require_backend


def _calibration_geometry(payload):
    cameras=payload.get("cameras")
    points=payload.get("points3d")
    if cameras and points:
        return cameras,points
    cameras=payload.get("scaled_cameras")
    points=payload.get("scaled_points3d")
    if cameras and points:
        return cameras,points
    raise ValueError("surface calibration must contain one coherent cameras/points3d reconstruction")


def _reprojection_diagnostics(cameras,xyz,point_indices,cam_indices,observed_uv):
    import cv2
    errors=[]
    for camera_index,camera in enumerate(cameras):
        selected=np.flatnonzero(cam_indices==camera_index)
        if not len(selected): continue
        rotation=np.asarray(camera["R"],np.float64)
        translation=np.asarray(camera["t"],np.float64).reshape(3,1)
        intrinsic=np.asarray(camera["K"],np.float64)
        distortion=np.asarray(camera.get("distortion",[]),np.float64)
        projected,_=cv2.projectPoints(xyz[point_indices[selected]].reshape(-1,1,3),cv2.Rodrigues(rotation)[0],translation,intrinsic,distortion)
        errors.append(np.linalg.norm(projected.reshape(-1,2)-observed_uv[selected],axis=1))
    error=np.concatenate(errors)
    return {"mean":float(error.mean()),"median":float(np.median(error)),"p95":float(np.percentile(error,95))}


def _mad_upper_mask(values, threshold):
    values=np.asarray(values,np.float64); median=float(np.median(values)); mad=float(np.median(np.abs(values-median)))
    if mad<=1e-12 or threshold<=0:return np.ones(len(values),bool)
    return values<=median+float(threshold)*1.4826*mad


def _filter_sparse_points(points, xyz, config):
    """Match NDeF-DIC's track/reprojection/radius/KNN sparse filter."""
    from scipy.spatial import cKDTree
    count=len(xyz);keep=np.isfinite(xyz).all(axis=1);reasons={"nonfinite":int((~keep).sum())}
    track=np.asarray([len(point.get("observations",[])) for point in points],np.int64)
    track_ok=track>=int(config.get("min_track_length",2));keep&=track_ok;reasons["short_track"]=int((~track_ok).sum())
    maximum=config.get("max_reprojection_error",3.0)
    reprojection=np.asarray([point.get("reprojection_error",np.inf) for point in points],np.float64)
    if maximum is not None:
        reproj_ok=np.isfinite(reprojection)&(reprojection<=float(maximum));keep&=reproj_ok;reasons["high_reprojection_error"]=int((~reproj_ok).sum())
    centre=np.median(xyz,axis=0);radius=np.linalg.norm(xyz-centre,axis=1);radius_ok=_mad_upper_mask(radius,float(config.get("radius_mad_thresh",8.0)));keep&=radius_ok;reasons["robust_radius"]=int((~radius_ok).sum())
    k=int(config.get("knn_k",8));knn_ok=np.ones(count,bool)
    if count>=max(8,k+2) and k>0:
        distance,_=cKDTree(xyz).query(xyz,k=min(k+1,count));knn_ok=_mad_upper_mask(distance[:,-1],float(config.get("knn_mad_thresh",8.0)))
    keep&=knn_ok;reasons["knn_density"]=int((~knn_ok).sum())
    return keep,{"n_points_before":count,"n_points_after":int(keep.sum()),"criteria":dict(config),"raw_rejection_counts":reasons}


def _save_dense_surface_visualizations(vis,names,masks,dense_uv,dense_camera,dense_depth,dense_world):
    import matplotlib.pyplot as plt
    world=np.asarray(dense_world,np.float32)
    radius=np.linalg.norm(world[:,[0,2]],axis=1)
    stride=max(1,len(world)//100000)
    shown=world[::stride]
    shown_radius=radius[::stride]
    fig=plt.figure(figsize=(8,7),constrained_layout=True)
    axis=fig.add_subplot(projection="3d")
    points=axis.scatter(shown[:,0],shown[:,1],shown[:,2],c=shown_radius,s=1,cmap="viridis")
    axis.set(xlabel="world X",ylabel="world Y",zlabel="world Z",title="Dense reconstructed reference surface")
    fig.colorbar(points,ax=axis,label="radial distance")
    # This is the union of source-camera charts before multi-view fusion.
    # Keep it for diagnostics; dense_world_surface.png is reserved for the
    # fused deformation hand-off surface below.
    fig.savefig(vis/"dense_world_raw_charts.png",dpi=180)
    plt.close(fig)
    for camera_index,name in enumerate(names):
        selected=np.asarray(dense_camera)==camera_index
        if not np.any(selected): continue
        uv=np.rint(np.asarray(dense_uv)[selected]).astype(np.int64)
        depth=np.asarray(dense_depth)[selected]
        image=np.full(masks[camera_index].shape,np.nan,np.float32)
        image[uv[:,1],uv[:,0]]=depth
        fig,axis=plt.subplots(figsize=(6,5),constrained_layout=True)
        rendered=axis.imshow(image,cmap="viridis")
        axis.set(title=f"{name} dense reconstructed Z-depth",xlabel="u",ylabel="v")
        fig.colorbar(rendered,ax=axis,label="camera Z-depth")
        fig.savefig(vis/f"{name}_dense_depth.png",dpi=160)
        plt.close(fig)


def _sample_stride_depth(depth_grid,uv,stride):
    height,width=depth_grid.shape
    xy=np.asarray(uv,np.float64)/float(stride)
    x0=np.floor(xy[:,0]).astype(np.int64); y0=np.floor(xy[:,1]).astype(np.int64); x1=x0+1; y1=y0+1
    valid=(x0>=0)&(y0>=0)&(x1<width)&(y1<height)
    out=np.full(len(uv),np.nan,np.float32)
    idx=np.flatnonzero(valid)
    if not len(idx): return out
    q00=depth_grid[y0[idx],x0[idx]];q10=depth_grid[y0[idx],x1[idx]];q01=depth_grid[y1[idx],x0[idx]];q11=depth_grid[y1[idx],x1[idx]]
    finite=np.isfinite(q00)&np.isfinite(q10)&np.isfinite(q01)&np.isfinite(q11)
    idx=idx[finite]
    if not len(idx): return out
    dx=xy[idx,0]-x0[idx];dy=xy[idx,1]-y0[idx]
    out[idx]=q00[finite]*(1-dx)*(1-dy)+q10[finite]*dx*(1-dy)+q01[finite]*(1-dx)*dy+q11[finite]*dx*dy
    return out


def _estimate_surface_normals_from_dense(points,dense_points):
    from scipy.spatial import cKDTree
    tree=cKDTree(dense_points); neighbours=tree.query(points,k=min(24,len(dense_points)),workers=-1)[1]; normals=np.empty_like(points)
    for index,neighbour_ids in enumerate(neighbours):
        local=dense_points[np.atleast_1d(neighbour_ids)]; centered=local-local.mean(axis=0); _,vectors=np.linalg.eigh(centered.T@centered/max(1,len(local)-1)); normals[index]=vectors[:,0]
    return normals/np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)


def _surface_maps(field_uv,field_camera,field_depth,field_world,masks):
    maps=[]
    for camera_index,mask in enumerate(masks):
        depth=np.full(mask.shape,np.nan,np.float32);world=np.full((*mask.shape,3),np.nan,np.float32);selected=field_camera==camera_index;pix=np.rint(field_uv[selected]).astype(np.int64);depth[pix[:,1],pix[:,0]]=field_depth[selected];world[pix[:,1],pix[:,0]]=field_world[selected]
        valid=np.isfinite(depth);area=np.zeros(mask.shape,np.float64);inner=valid[1:-1,1:-1]&valid[1:-1,:-2]&valid[1:-1,2:]&valid[:-2,1:-1]&valid[2:,1:-1]
        xu=world[1:-1,2:]-world[1:-1,:-2];xv=world[2:,1:-1]-world[:-2,1:-1];local=.25*np.linalg.norm(np.cross(xu,xv),axis=2);area[1:-1,1:-1]=np.where(inner&np.isfinite(local),local,0)
        if not np.any(area) and np.any(valid):area[valid]=1
        maps.append({"depth":depth,"world":world,"area":area})
    return maps


def _area_weighted_candidates(maps,spacing,config,rng):
    candidate_spacing=spacing*float(config.get("fusion_candidate_spacing_factor",.5));target_area=candidate_spacing**2;requested=[]
    for item in maps:
        flat=item["area"].ravel();valid=np.flatnonzero(flat>0);requested.append(min(len(valid),int(np.ceil(flat[valid].sum()/target_area)) if len(valid) else 0))
    maximum=int(config.get("fusion_max_candidate_points",1200000));total=sum(requested)
    if total>maximum:requested=[max(1,int(np.floor(n*maximum/total))) if n else 0 for n in requested]
    points=[];source=[]
    for camera_index,(item,count) in enumerate(zip(maps,requested)):
        flat=item["area"].ravel();valid=np.flatnonzero(flat>0)
        if not count or not len(valid):continue
        weights=flat[valid];weights/=weights.sum();chosen=rng.choice(valid,size=min(count,len(valid)),replace=False,p=weights);y,x=np.unravel_index(chosen,item["area"].shape);candidate=item["world"][y,x];finite=np.isfinite(candidate).all(axis=1);points.append(candidate[finite]);source.append(np.full(finite.sum(),camera_index,np.int16))
    if not points:raise RuntimeError("dense surface fusion found no area-supported 2.5D candidates")
    return np.concatenate(points),np.concatenate(source)


def _farthest_point_indices(points,count,rng):
    selected=np.empty(count,np.int64);selected[0]=int(rng.integers(0,len(points)));minimum=np.full(len(points),np.inf,np.float64)
    for index in range(1,count):
        delta=points-points[selected[index-1]];minimum=np.minimum(minimum,np.einsum("ij,ij->i",delta,delta));selected[index]=int(np.argmax(minimum))
    return selected


def _fuse_dense_surface(field_uv,field_camera,field_depth,field_world,masks,cameras,stride,config):
    import cv2
    uv=np.asarray(field_uv,np.float32);source_field=np.asarray(field_camera,np.int16);depth_field=np.asarray(field_depth,np.float32);dense_points=np.asarray(field_world,np.float32);rng=np.random.default_rng(int(config.get("fusion_seed",17)))
    low=np.percentile(dense_points,1,axis=0);high=np.percentile(dense_points,99,axis=0);object_scale=float(np.linalg.norm(high-low));spacing=float(config.get("fusion_relative_sample_spacing",0.006)*object_scale);depth_range=float(np.percentile(depth_field,99)-np.percentile(depth_field,1));tolerance=max(spacing*float(config.get("fusion_depth_tolerance_factor",1.0)),depth_range*1e-3)
    maps=_surface_maps(uv,source_field,depth_field,dense_points,masks);points,source=_area_weighted_candidates(maps,spacing,config,rng);camera_count=len(cameras);grids=[item["depth"] for item in maps]
    visible=np.zeros((len(points),camera_count),bool);errors=np.full((len(points),camera_count),np.nan,np.float32);projected=np.full((len(points),camera_count,2),np.nan,np.float32);projected_depth=np.full((len(points),camera_count),np.nan,np.float32)
    for camera_index,camera in enumerate(cameras):
        rotation=np.asarray(camera["R"],np.float64);translation=np.asarray(camera["t"],np.float64).reshape(3,1);intrinsic=np.asarray(camera["K"],np.float64);distortion=np.asarray(camera.get("distortion",[]),np.float64)
        target_uv,_=cv2.projectPoints(points.reshape(-1,1,3),cv2.Rodrigues(rotation)[0],translation,intrinsic,distortion);target_uv=target_uv.reshape(-1,2);target_depth=(rotation@points.T+translation)[2];projected[:,camera_index]=target_uv;projected_depth[:,camera_index]=target_depth
        mask=masks[camera_index]; pix=np.rint(target_uv).astype(np.int64); inside=(target_depth>1e-8)&(pix[:,0]>=0)&(pix[:,0]<mask.shape[1])&(pix[:,1]>=0)&(pix[:,1]<mask.shape[0]); in_roi=np.zeros(len(points),bool); valid=np.flatnonzero(inside); in_roi[valid]=mask[pix[valid,1],pix[valid,0]]
        reference=_sample_stride_depth(grids[camera_index],target_uv,1);error=np.abs(target_depth-reference);errors[:,camera_index]=error;visible[:,camera_index]=in_roi&np.isfinite(error)&(error<=tolerance)
    min_visible=int(config.get("fusion_min_visible_cameras",2)); counts=visible.sum(axis=1); keep=counts>=min_visible; candidate=np.flatnonzero(keep)
    if not len(candidate): raise RuntimeError("dense surface fusion found no multi-view depth-consistent points")
    mean_error=np.nanmean(np.where(visible[candidate],errors[candidate],np.nan),axis=1);voxel=np.floor(points[candidate]/spacing).astype(np.int64);order=np.lexsort((mean_error,-counts[candidate]));_,first=np.unique(voxel[order],axis=0,return_index=True);selected=candidate[order[first]]
    maximum=int(config.get("fusion_max_points",100000))
    if len(selected)>maximum:selected=selected[_farthest_point_indices(points[selected],maximum,rng)]
    fused=points[selected];normals=_estimate_surface_normals_from_dense(fused,dense_points);return {"points":fused,"normals":normals,"source_camera":source[selected],"visibility_mask":visible[selected],"projected_uv":projected[selected],"projected_depth":projected_depth[selected],"depth_abs_error":errors[selected],"visible_counts":counts[selected].astype(np.int16),"object_scale":object_scale,"sample_spacing":spacing,"depth_tolerance":tolerance,"chart_candidate_count":int(len(points)),"candidate_count":int(len(candidate))}


def _save_fused_surface_visualization(path,fusion,colour_by="visibility"):
    import matplotlib.pyplot as plt
    points=fusion["points"]
    if colour_by=="radial_distance":
        colours=np.linalg.norm(points[:,[0,2]],axis=1); label="radial distance"; title="Visibility/depth-consistency fused reference surface"
    else:
        colours=fusion["visible_counts"]; label="consistent visible cameras"; title="Visibility/depth-consistency fused reference surface"
    fig=plt.figure(figsize=(8,7),constrained_layout=True); axis=fig.add_subplot(projection="3d"); plot=axis.scatter(points[:,0],points[:,1],points[:,2],c=colours,s=1,cmap="viridis"); axis.set(xlabel="world X",ylabel="world Y",zlabel="world Z",title=title); fig.colorbar(plot,ax=axis,label=label);fig.savefig(path,dpi=180);plt.close(fig)

def pretrain_ndef_surface(config="config/ndef_multi.yaml"):
    values=load_config(config) if isinstance(config,(str,Path)) else config; case=values["case"]; root=Path(case["root"]); cal=root/case["calibration"]
    payload=json.loads(cal.read_text()); cams,points=_calibration_geometry(payload); names=[camera_name_from_label(x.get("label", ""),f"cam_{index}") for index,x in enumerate(cams)]
    obs=np.load(root/"result/calibration/observations.npz"); xyz=np.asarray([p["xyz"] for p in points],np.float32)
    ids=obs["point_indices"].astype(int); ci=obs["cam_indices"].astype(int); uv=obs["uv"].astype(np.float32); R=np.asarray([x["R"] for x in cams],np.float32); t=np.asarray([x["t"] for x in cams],np.float32)
    sparse_filter_cfg=values.get("surface",{}).get("sparse_filter",{}); point_keep,sparse_filter=_filter_sparse_points(points,xyz,sparse_filter_cfg); observation_sparse_keep=point_keep[ids]
    reprojection=_reprojection_diagnostics(cams,xyz,ids,ci,uv)
    max_p95=float(values.get("surface",{}).get("max_reprojection_p95_px",5.0))
    if reprojection["p95"]>max_p95: raise ValueError(f"surface calibration cameras/points mismatch: reprojection p95={reprojection['p95']:.3f}px > {max_p95:.3f}px")
    depth=np.einsum("nij,nj->ni",R[ci],xyz[ids])+t[ci]; depth=depth[:,2]
    mask_path=case.get("masks")
    if mask_path is None:
        mask_root,_=ndef_run_roots(root,values); mask_root=mask_root/"roi"/"per_camera"
    else:
        mask_root=Path(mask_path); mask_root=mask_root if mask_root.is_absolute() else root/mask_root
    masks=[]; queries=[]; query_c=[]; stride=1
    for k,n in enumerate(names):
        m=np.load(mask_root/f"{n}_mask.npy").astype(bool); masks.append(m); yy,xx=np.where(m[::stride,::stride]); queries.append(np.c_[xx*stride,yy*stride]);query_c.append(np.full(len(xx),k))
    roi_support=np.zeros(len(uv),dtype=bool)
    rounded=np.rint(uv).astype(np.int64)
    for camera_index,mask in enumerate(masks):
        selected=np.flatnonzero(ci==camera_index); x=rounded[selected,0]; y=rounded[selected,1]
        inside=(x>=0)&(x<mask.shape[1])&(y>=0)&(y<mask.shape[0]); valid=selected[inside]
        roi_support[valid]=mask[y[inside],x[inside]]
    positive_roi=roi_support&observation_sparse_keep&(depth>0)
    if not np.any(positive_roi): raise ValueError("surface sparse supervision has no positive-depth observations inside the per-camera ROIs")
    med=np.median(depth[positive_roi]); mad=np.median(np.abs(depth[positive_roi]-med)); depth_clean=(depth>0)&(np.abs(depth-med)<=6*max(mad,1e-6)); keep=roi_support&observation_sparse_keep&depth_clean
    backend=_require_backend(); sizes=np.asarray([[x["image_width"],x["image_height"]] for x in cams],np.float32); q=np.concatenate(queries).astype(np.float32); qc=np.concatenate(query_c).astype(np.int64)
    p=backend.NDeFSurfaceProblem(torch.from_numpy(uv[keep]),torch.from_numpy(ci[keep]),torch.from_numpy(depth[keep].astype(np.float32)),torch.from_numpy(sizes),torch.from_numpy(np.stack(masks)),torch.from_numpy(q),torch.from_numpy(qc)); cfg=values.get("surface_training",{}); model=values.get("surface_model",{}); o=backend.NDeFDepthModelOptions()
    for k,v in model.items():
        if hasattr(o,k): setattr(o,k,v)
    p.model_options=o
    for k in ("pretrain_iterations","pretrain_learning_rate","weight_decay","smoothness_weight","smooth_samples_per_camera"):
        if k in cfg:setattr(p,k,cfg[k])
    # Dense refinement remains a thin assembly concern here.  The ROI-spacing
    # sampler, pair filtering, geometry, ZNSSD and optimization all run in C++.
    dense=values.get("surface_dense_training", {})
    dense_enabled=bool(dense.get("enabled",int(dense.get("iterations",0))>0))
    if dense_enabled:
        import cv2
        pair_data=json.loads((root/"result/calibration/camera_pairs.json").read_text(encoding="utf-8"))
        pair_names=pair_data["camera_names"]
        if pair_names != names: raise ValueError("camera_pairs.json camera_names must match scaled calibration order")
        neighbors=np.full((len(names),2),-1,np.int64)
        name_to_id={name:index for index,name in enumerate(names)}
        for index,name in enumerate(names):
            listed=pair_data["neighbors"].get(name,[])
            if len(listed)>2: raise ValueError(f"{name} has more than two topology neighbours")
            neighbors[index,:len(listed)]=[name_to_id[target] for target in listed]
        images=[]
        for name in names:
            image=cv2.imread(str(root/case["images"]/name/"001.bmp"),cv2.IMREAD_GRAYSCALE)
            if image is None: raise FileNotFoundError(root/case["images"]/name/"001.bmp")
            images.append(image.astype(np.float32)/255.0)
        images=np.stack(images)
        K=np.asarray([x["K"] for x in cams],np.float32); dist=np.asarray([x.get("distortion",[]) for x in cams],np.float32)
        p.set_dense_inputs(torch.from_numpy(images),torch.from_numpy(K),torch.from_numpy(R),torch.from_numpy(t),torch.from_numpy(dist),torch.from_numpy(neighbors))
        p.dense_iterations=1
        mapping={"epochs":"dense_epochs","samples_per_camera":"dense_samples_per_camera","auto_batch":"dense_auto_batch","auto_batch_start":"dense_auto_batch_start","memory_fraction":"dense_memory_fraction","spacing_px":"dense_spacing_px","patch_radius":"dense_patch_radius","learning_rate":"dense_learning_rate","anchor_weight":"dense_anchor_weight","min_valid_patch_ratio":"dense_min_valid_patch_ratio","seed":"dense_seed","prediction_batch_size":"prediction_batch_size"}
        for key,attribute in mapping.items():
            if key in dense: setattr(p,attribute,dense[key])
    p.set_device(cfg.get("device","cuda")); r=backend.NDeFSurfaceSolver().solve(p)
    run_out,run_vis=ndef_run_roots(root,values)
    pretrain_out=run_out/"pretrain"/"surface"; pretrain_vis=run_vis/"pretrain"/"surface"; pretrain_out.mkdir(parents=True,exist_ok=True);pretrain_vis.mkdir(parents=True,exist_ok=True)
    out=run_out/"surface"; vis=run_vis/"surface"; out.mkdir(parents=True,exist_ok=True);vis.mkdir(parents=True,exist_ok=True)
    roi_uv_bounds=np.asarray([[np.where(mask)[1].min(),np.where(mask)[0].min(),np.where(mask)[1].max(),np.where(mask)[0].max()] for mask in masks],np.float32)
    np.savez_compressed(pretrain_out/"surface_pretrain.npz",sparse_uv=uv[keep],sparse_camera=ci[keep],sparse_depth=depth[keep],sparse_prediction=r.sparse_prediction.numpy(),query_uv=r.query_uv.numpy(),query_camera=r.query_cameras.numpy(),query_depth=r.query_depth.numpy(),roi_uv_bounds=roi_uv_bounds,depth_mean=np.asarray(r.depth_mean),depth_std=np.asarray(r.depth_std))
    (pretrain_out/"surface_pretrain_meta.json").write_text(json.dumps({"kept_sparse_observations":int(keep.sum()),"rejected_by_sparse_filter":int((~observation_sparse_keep).sum()),"sparse_filter":sparse_filter,"rejected_outside_roi":int((~roi_support).sum()),"rejected_depth_outliers":int((roi_support&observation_sparse_keep&(~depth_clean)).sum()),"depth_median":float(med),"depth_mad":float(mad),"depth_normalization":{"mean":float(r.depth_mean),"std":float(r.depth_std)},"pixel_normalization":"full_image_width_height","calibration_reprojection_px":reprojection,"diagnostics":{"sparse_rmse":float(r.diagnostics.metrics["sparse_rmse"])}},indent=2))
    if r.dense_uv is not None and r.dense_uv.numel() > 0:
        np.savez_compressed(out/"surface_dense_samples.npz",uv=r.dense_uv.numpy(),camera=r.dense_cameras.numpy(),targets=r.dense_targets.numpy(),depth=r.dense_depth.numpy(),world=r.dense_world.numpy(),history=r.dense_history.numpy(),history_columns=np.asarray(["photo_loss","anchor_loss","total_loss"]),roi_uv_bounds=roi_uv_bounds,depth_mean=np.asarray(r.depth_mean),depth_std=np.asarray(r.depth_std))
        np.savez_compressed(out/"surface_dense_field.npz",uv=r.dense_field_uv.numpy(),camera=r.dense_field_cameras.numpy(),depth=r.dense_field_depth.numpy(),world=r.dense_field_world.numpy(),grid_stride=np.asarray(stride),roi_uv_bounds=roi_uv_bounds,depth_mean=np.asarray(r.depth_mean),depth_std=np.asarray(r.depth_std))
        fusion=_fuse_dense_surface(r.dense_field_uv.numpy(),r.dense_field_cameras.numpy(),r.dense_field_depth.numpy(),r.dense_field_world.numpy(),masks,cams,stride,values.get("surface",{}))
        np.savez_compressed(out/"deformation_surface_dataset.npz",points=fusion["points"],normals=fusion["normals"],source_camera=fusion["source_camera"],visibility_mask=fusion["visibility_mask"],projected_uv=fusion["projected_uv"],projected_depth=fusion["projected_depth"],depth_abs_error=fusion["depth_abs_error"],visible_counts=fusion["visible_counts"],cam_names=np.asarray(names))
        # Keep the same standalone surface products as the reference
        # reconstruction_dense.py -> surface_sampler.py route, while the NPZ
        # remains the compact hand-off contract for the deformation stage.
        np.save(out/"surface_points.npy",fusion["points"])
        np.save(out/"surface_normals.npy",fusion["normals"])
        np.save(out/"source_camera.npy",fusion["source_camera"])
        np.save(out/"visibility_mask.npy",fusion["visibility_mask"])
        np.save(out/"projected_uv.npy",fusion["projected_uv"])
        np.save(out/"projected_depth.npy",fusion["projected_depth"])
        _save_fused_surface_visualization(vis/"fused_surface.png",fusion)
        _save_fused_surface_visualization(vis/"dense_world_surface.png",fusion,colour_by="radial_distance")
        (out/"surface_dense_meta.json").write_text(json.dumps({"config":dense,"normalization":{"pixel_coordinates":"full_image_width_height","depth_mean":float(r.depth_mean),"depth_std":float(r.depth_std)},"calibration_reprojection_px":reprojection,"diagnostics":dict(r.diagnostics.metrics),"history_steps":int(r.dense_history.shape[0]),"history_columns":["photo_loss","anchor_loss","total_loss"],"fusion":{"method":"area_weighted_2p5d_depth_consistency","relative_sample_spacing":float(values.get("surface",{}).get("fusion_relative_sample_spacing",.006)),"sample_spacing":float(fusion["sample_spacing"]),"depth_tolerance":float(fusion["depth_tolerance"]),"min_visible_cameras":int(values.get("surface",{}).get("fusion_min_visible_cameras",2)),"chart_candidate_count":int(fusion["chart_candidate_count"]),"candidate_count":int(fusion["candidate_count"]),"surface_point_count":int(len(fusion["points"]))}},indent=2))
        _save_dense_surface_visualizations(vis,names,masks,r.dense_uv.numpy(),r.dense_cameras.numpy(),r.dense_depth.numpy(),r.dense_world.numpy())
        import matplotlib.pyplot as plt
        h=np.asarray(r.dense_history.numpy(),np.float64)
        steps=np.arange(1,len(h)+1)
        fig,ax=plt.subplots();ax.plot(steps,h[:,0]);ax.set(xlabel="Dense refinement step",ylabel="ZNSSD photo loss",title="NDeF surface dense ZNSSD optimisation");ax.grid(alpha=.25);fig.savefig(vis/"dense_photo_loss.png",dpi=140);plt.close(fig)
        fig,ax=plt.subplots();ax.plot(steps,h[:,0],label="photo_loss");ax.plot(steps,h[:,1],label="anchor_loss");ax.plot(steps,h[:,2],label="total_loss");ax.set(xlabel="Dense refinement step",ylabel="loss",title="NDeF surface dense loss components");ax.grid(alpha=.25);ax.legend();fig.savefig(vis/"dense_loss_components.png",dpi=140);plt.close(fig)
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    predicted_query=r.query_depth.numpy()
    for k,n in enumerate(names):
        sel=qc==k; uvq=q[sel]; yy,xx=np.where(masks[k]); grid=np.full(masks[k].shape,np.nan,np.float32)
        # The compiled model is queried on a bounded-stride ROI grid for memory
        # control.  Interpolate those continuous-field samples back to every ROI
        # pixel before visualising so it is comparable to the SfM depth image.
        grid[yy,xx]=griddata(uvq,predicted_query[sel],np.c_[xx,yy],method="linear")
        if np.isnan(grid[yy,xx]).any():
            missing=np.isnan(grid[yy,xx]); nearest=griddata(uvq,predicted_query[sel],np.c_[xx[missing],yy[missing]],method="nearest"); grid[yy[missing],xx[missing]]=nearest
        sparse=(ci[keep]==k); interp=griddata(uv[keep][sparse],depth[keep][sparse],np.c_[xx,yy],method="linear"); sfm_grid=np.full(masks[k].shape,np.nan,np.float32);sfm_grid[yy,xx]=interp
        fig,ax=plt.subplots();im=ax.imshow(grid);fig.colorbar(im,ax=ax);ax.set_title(f"{n} sparse-pretrain predicted depth");fig.savefig(pretrain_vis/f"{n}_predicted_depth.png",dpi=140);plt.close(fig)
        fig,ax=plt.subplots(1,2,figsize=(10,4),constrained_layout=True);lo=np.nanmin([grid,sfm_grid]);hi=np.nanmax([grid,sfm_grid]);a=ax[0].imshow(grid,vmin=lo,vmax=hi);ax[0].set_title(f"{n} sparse-pretrain predicted depth");ax[1].imshow(sfm_grid,vmin=lo,vmax=hi);ax[1].set_title(f"{n} SfM depth interpolation");fig.colorbar(a,ax=ax);fig.savefig(pretrain_vis/f"{n}_depth_overview.png",dpi=140);plt.close(fig)
    fig=plt.figure();ax=fig.add_subplot(projection="3d");ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],s=1,c=xyz[:,2]);fig.savefig(pretrain_vis/"sfm_sparse_surface.png",dpi=140);plt.close(fig)
    return r
