#include "neurodic/solver/ndef_surface_solver.hpp"

#include <ATen/Context.h>
#include <cmath>
#include <torch/cuda.h>
#include <vector>

#include "neurodic/core/exceptions.hpp"
#include "neurodic/model/ndef_depth_model.hpp"

namespace neurodic {
namespace {
torch::Tensor roi_bounds(const torch::Tensor& masks,const torch::Device& device) {
    std::vector<float> values;values.reserve(static_cast<size_t>(masks.size(0))*4);
    for(int64_t camera=0;camera<masks.size(0);++camera){auto points=torch::nonzero(masks.index({camera}));if(points.numel()==0) throw ValidationError("NDeF surface ROI mask is empty");auto minimum=std::get<0>(points.min(0)),maximum=std::get<0>(points.max(0));values.push_back(minimum.index({1}).item<float>());values.push_back(minimum.index({0}).item<float>());values.push_back(maximum.index({1}).item<float>());values.push_back(maximum.index({0}).item<float>());}
    return torch::from_blob(values.data(),{masks.size(0),4},torch::TensorOptions().dtype(torch::kFloat32)).clone().to(device);
}
torch::Tensor norm_uv(const torch::Tensor& uv, const torch::Tensor& cameras, const torch::Tensor& bounds) {
    auto selected=bounds.index_select(0,cameras);auto minimum=selected.slice(1,0,2),maximum=selected.slice(1,2,4);
    return 2.0F*(uv-minimum)/(maximum-minimum).clamp_min(1.0F)-1.0F;
}
torch::Tensor patch_inside_image(const torch::Tensor& uv,const torch::Tensor& cameras,const torch::Tensor& sizes,int radius) {
    auto wh=sizes.index_select(0,cameras);auto x=uv.select(1,0),y=uv.select(1,1);auto r=static_cast<float>(radius);
    return (x>=r)&(x<=wh.select(1,0)-1.0F-r)&(y>=r)&(y<=wh.select(1,1)-1.0F-r);
}
torch::Tensor offsets(int radius, const torch::TensorOptions& options) {
    auto a=torch::arange(-radius,radius+1,options); auto n=2*radius+1;
    return torch::stack({a.repeat({n}),a.repeat_interleave(n)},1);
}
torch::Tensor mask_at(const torch::Tensor& masks, const torch::Tensor& uv, const torch::Tensor& cameras) {
    const auto h=masks.size(1), w=masks.size(2); auto x=uv.select(2,0),y=uv.select(2,1);
    auto inside=(x>=0)&(x<w)&(y>=0)&(y<h);
    auto xi=x.clamp(0,w-1).to(torch::kLong), yi=y.clamp(0,h-1).to(torch::kLong);
    auto ci=cameras.unsqueeze(1).expand_as(xi);
    return inside & masks.index({ci,yi,xi});
}
torch::Tensor sample_bilinear(const torch::Tensor& images, const torch::Tensor& uv, const torch::Tensor& cameras) {
    const auto h=images.size(1),w=images.size(2); auto x=uv.select(2,0),y=uv.select(2,1);
    auto x0=torch::floor(x).to(torch::kLong),y0=torch::floor(y).to(torch::kLong); auto x1=x0+1,y1=y0+1;
    auto ci=cameras.unsqueeze(1).expand_as(x0);
    auto at=[&](const torch::Tensor& xx,const torch::Tensor& yy){return images.index({ci,yy.clamp(0,h-1),xx.clamp(0,w-1)});};
    auto x0f=x0.to(x.scalar_type()),y0f=y0.to(y.scalar_type());
    return (x1.to(x.scalar_type())-x)*(y1.to(y.scalar_type())-y)*at(x0,y0)+
           (x1.to(x.scalar_type())-x)*(y-y0f)*at(x0,y1)+
           (x-x0f)*(y1.to(y.scalar_type())-y)*at(x1,y0)+
           (x-x0f)*(y-y0f)*at(x1,y1);
}
torch::Tensor undistort(torch::Tensor xd, const torch::Tensor& d) {
    auto xy=xd;
    auto c=[&](int i){return i<d.size(1)?d.select(1,i):torch::zeros({d.size(0)},d.options());};
    for(int n=0;n<5;++n){auto x=xy.select(1,0),y=xy.select(1,1),r2=x*x+y*y;auto radial=1+c(0)*r2+c(1)*r2*r2+c(4)*r2*r2*r2;
        auto dx=2*c(2)*x*y+c(3)*(r2+2*x*x),dy=c(2)*(r2+2*y*y)+2*c(3)*x*y;
        xy=torch::stack({(xd.select(1,0)-dx)/radial,(xd.select(1,1)-dy)/radial},1);}
    return xy;
}
torch::Tensor backproject(const torch::Tensor& uv,const torch::Tensor& depth,const torch::Tensor& cams,
                          const torch::Tensor& k,const torch::Tensor& r,const torch::Tensor& t,const torch::Tensor& d) {
    auto ki=k.index_select(0,cams), di=d.index_select(0,cams), ri=r.index_select(0,cams), ti=t.index_select(0,cams);
    auto xy=undistort(torch::stack({(uv.select(1,0)-ki.select(1,0).select(1,2))/ki.select(1,0).select(1,0),
                                     (uv.select(1,1)-ki.select(1,1).select(1,2))/ki.select(1,1).select(1,1)},1),di);
    auto camera=torch::stack({xy.select(1,0)*depth,xy.select(1,1)*depth,depth},1);
    return torch::bmm(ri.transpose(1,2),(camera-ti).unsqueeze(2)).squeeze(2);
}
std::pair<torch::Tensor,torch::Tensor> project(const torch::Tensor& world,const torch::Tensor& cams,
    const torch::Tensor& k,const torch::Tensor& r,const torch::Tensor& t,const torch::Tensor& d) {
    auto ki=k.index_select(0,cams),di=d.index_select(0,cams),ri=r.index_select(0,cams),ti=t.index_select(0,cams);
    auto q=torch::bmm(ri,world.unsqueeze(2)).squeeze(2)+ti; auto z=q.select(1,2); auto x=q.select(1,0)/z.clamp_min(1e-8F),y=q.select(1,1)/z.clamp_min(1e-8F);
    auto c=[&](int i){return i<di.size(1)?di.select(1,i):torch::zeros_like(z);}; auto r2=x*x+y*y; auto radial=1+c(0)*r2+c(1)*r2*r2+c(4)*r2*r2*r2;
    auto xd=x*radial+2*c(2)*x*y+c(3)*(r2+2*x*x),yd=y*radial+c(2)*(r2+2*y*y)+2*c(3)*x*y;
    return {torch::stack({ki.select(1,0).select(1,0)*xd+ki.select(1,0).select(1,2),ki.select(1,1).select(1,1)*yd+ki.select(1,1).select(1,2)},1),z};
}
torch::Tensor patch_znssd(const torch::Tensor& a,const torch::Tensor& b,const torch::Tensor& valid) {
    auto w=valid.to(a.scalar_type()),n=w.sum(1,true).clamp_min(1);auto ma=(w*a).sum(1,true)/n,mb=(w*b).sum(1,true)/n;
    auto sa=torch::sqrt((w*(a-ma).square()).sum(1,true)/n+1e-6F),sb=torch::sqrt((w*(b-mb).square()).sum(1,true)/n+1e-6F);
    return (w*((a-ma)/sa-(b-mb)/sb).square()).sum(1)/n.squeeze(1);
}
struct Samples { torch::Tensor uv,cams,targets,source_patch_valid; };
Samples build_samples(NDeFDepthModel& model,const NDeFSurfaceProblem& p,const torch::Tensor& mean,const torch::Tensor& sd,const torch::Tensor& bounds) {
    auto cpu=torch::TensorOptions().dtype(torch::kFloat32); std::vector<float> uvs;std::vector<int64_t> cams;
    auto masks=p.roi_masks; for(int64_t c=0;c<masks.size(0);++c) for(int64_t y=0;y<masks.size(1);y+=p.dense_spacing_px) for(int64_t x=0;x<masks.size(2);x+=p.dense_spacing_px) if(masks.index({c,y,x}).item<bool>()){uvs.push_back(float(x));uvs.push_back(float(y));cams.push_back(c);}
    if(cams.empty()) throw ValidationError("NDeF dense refinement found no ROI-spaced centres");
    auto uv=torch::from_blob(uvs.data(),{static_cast<int64_t>(cams.size()),2},cpu).clone().to(p.device);auto ci=torch::from_blob(cams.data(),{static_cast<int64_t>(cams.size())},torch::TensorOptions().dtype(torch::kLong)).clone().to(p.device);
    auto k=p.intrinsics.to(p.device),r=p.rotations.to(p.device),t=p.translations.to(p.device),d=p.distortions.to(p.device),neigh=p.dense_neighbors.to(p.device),m=p.roi_masks.to(p.device);
    torch::NoGradGuard ng; auto sizes=p.image_sizes.to(p.device),depth=model.forward(norm_uv(uv,ci,bounds),ci)*sd+mean;auto world=backproject(uv,depth,ci,k,r,t,d); auto off=offsets(p.dense_patch_radius,uv.options());auto source_valid=mask_at(m,uv.unsqueeze(1)+off.unsqueeze(0),ci);auto source_inside=patch_inside_image(uv,ci,sizes,p.dense_patch_radius);
    auto targets=torch::full({uv.size(0),2},-1,torch::TensorOptions().device(p.device).dtype(torch::kLong));
    for(int slot=0;slot<2;++slot){auto tc=neigh.index_select(0,ci).select(1,slot);auto safe=tc.clamp_min(0);auto q=project(world,safe,k,r,t,d);auto center=mask_at(m,q.first.unsqueeze(1),safe).squeeze(1)&patch_inside_image(q.first,safe,sizes,p.dense_patch_radius)&(q.second>1e-8F)&(tc>=0);targets.select(1,slot).copy_(torch::where(center,tc,torch::full_like(tc,-1)));}
    auto required=static_cast<int64_t>(std::ceil(source_valid.size(1)*p.dense_min_valid_patch_ratio));auto keep=(targets>=0).any(1)&source_inside&(source_valid.sum(1)>=required);return {uv.index({keep}),ci.index({keep}),targets.index({keep}),source_valid.index({keep})};
}
} // namespace

NDeFSurfaceResult NDeFSurfaceSolver::solve(const NDeFSurfaceProblem& p) const {
    p.validate(); auto d=p.device; auto uv=p.sparse_uv.to(d),c=p.sparse_cameras.to(d),z=p.sparse_depth.to(d),sizes=p.image_sizes.to(d),bounds=roi_bounds(p.roi_masks,d);
    // The sparse pretrain and dense phase share one model instance.  Seed both
    // generators before model construction so the dense audit has one fixed
    // C++ sparse-pretrain starting point instead of a new random field.
    torch::manual_seed(p.dense_seed);
    if (d.is_cuda()) torch::cuda::manual_seed_all(p.dense_seed);
    at::globalContext().setDeterministicAlgorithms(true, false);
    auto mean=z.mean(),sd=z.std(false).clamp_min(1e-6F),target=(z-mean)/sd; NDeFDepthModel model(sizes.size(0),p.model_options);model.to(d);
    torch::optim::AdamW opt(model.parameters(),torch::optim::AdamWOptions(p.pretrain_learning_rate).weight_decay(p.weight_decay)); double loss=0;
    for(int i=0;i<p.pretrain_iterations;++i){opt.zero_grad();auto l=torch::mse_loss(model.forward(norm_uv(uv,c,bounds),c),target);l.backward();opt.step();loss=l.item<double>();}
    NDeFSurfaceResult out; out.depth_mean=mean.item<double>();out.depth_std=sd.item<double>();
    // Preserve the sparse-SfM pretraining surface as an immutable stage-one
    // product. Dense refinement must not redefine its RMSE or visualisation.
    { torch::NoGradGuard ng; out.sparse_prediction=(model.forward(norm_uv(uv,c,bounds),c)*sd+mean).cpu();
      auto qc=p.query_cameras.to(d),qu=p.query_uv.to(d); out.query_depth=(model.forward(norm_uv(qu,qc,bounds),qc)*sd+mean).cpu();
      out.diagnostics.metrics["sparse_rmse"]=torch::sqrt(torch::mse_loss(out.sparse_prediction,p.sparse_depth)).item<double>(); }
    if(p.dense_iterations>0){
        auto s=build_samples(model,p,mean,sd,bounds); if(s.cams.numel()==0) throw ValidationError("NDeF dense refinement has no pair-supported samples");
        auto images=p.reference_images.to(d),k=p.intrinsics.to(d),r=p.rotations.to(d),t=p.translations.to(d),dist=p.distortions.to(d),sparse_uv=p.sparse_uv.to(d),sparse_c=p.sparse_cameras.to(d),sparse_z=p.sparse_depth.to(d);auto off=offsets(p.dense_patch_radius,uv.options());auto req=static_cast<int64_t>(std::ceil(off.size(0)*p.dense_min_valid_patch_ratio));
        torch::optim::AdamW dense_opt(model.parameters(),torch::optim::AdamWOptions(p.dense_learning_rate).weight_decay(p.weight_decay)); std::vector<float> history;torch::manual_seed(p.dense_seed);if(d.is_cuda()) torch::cuda::manual_seed_all(p.dense_seed);
        for(int step=0;step<p.dense_iterations;++step){std::vector<torch::Tensor> ids;for(int64_t view=0;view<sizes.size(0);++view){auto pool=torch::nonzero(s.cams==view).reshape({-1});if(pool.numel()==0) throw ValidationError("NDeF dense refinement has an empty source-camera shard");auto choose=pool.numel()>=p.dense_samples_per_camera?torch::randperm(pool.numel(),pool.options()).slice(0,0,p.dense_samples_per_camera):torch::randint(pool.numel(),{p.dense_samples_per_camera},pool.options());ids.push_back(pool.index_select(0,choose));}auto id=torch::cat(ids);auto buv=s.uv.index_select(0,id),bc=s.cams.index_select(0,id),bt=s.targets.index_select(0,id),bvalid=s.source_patch_valid.index_select(0,id);
            dense_opt.zero_grad();auto bz=model.forward(norm_uv(buv,bc,bounds),bc)*sd+mean;auto world=backproject(buv,bz,bc,k,r,t,dist);auto src_uv=buv.unsqueeze(1)+off.unsqueeze(0);auto ref=sample_bilinear(images,src_uv,bc);std::vector<torch::Tensor> edge_losses;
            for(int slot=0;slot<2;++slot){auto tc=bt.select(1,slot),present=tc>=0,safe=tc.clamp_min(0);auto q=project(world,safe,k,r,t,dist);auto tuv=q.first.unsqueeze(1)+off.unsqueeze(0);auto valid=bvalid & mask_at(p.roi_masks.to(d),tuv,safe)&(q.second.unsqueeze(1)>1e-8F);auto good=present&(valid.sum(1)>=req);auto ids=torch::nonzero(good).reshape({-1});if(ids.numel()==0) continue;auto edge_photo=patch_znssd(ref.index_select(0,ids),sample_bilinear(images,tuv,safe).index_select(0,ids),valid.index_select(0,ids));edge_losses.push_back(edge_photo);}
            if(edge_losses.empty()) { throw ValidationError("NDeF dense refinement batch has no valid target patches"); } auto photo=torch::cat(edge_losses,0).mean();auto anchor=torch::mse_loss(model.forward(norm_uv(sparse_uv,sparse_c,bounds),sparse_c),target);auto total=photo+p.dense_anchor_weight*anchor;total.backward();dense_opt.step();history.push_back(photo.detach().item<float>());history.push_back(anchor.detach().item<float>());history.push_back(total.detach().item<float>());loss=total.item<double>();}
        torch::NoGradGuard ng;auto final_sparse=model.forward(norm_uv(sparse_uv,sparse_c,bounds),sparse_c)*sd+mean;auto candidate_depth=model.forward(norm_uv(s.uv,s.cams,bounds),s.cams)*sd+mean;auto field_uv=p.query_uv.to(d),field_cameras=p.query_cameras.to(d),field_depth=model.forward(norm_uv(field_uv,field_cameras,bounds),field_cameras)*sd+mean;out.dense_uv=s.uv.cpu();out.dense_cameras=s.cams.cpu();out.dense_targets=s.targets.cpu();out.dense_depth=candidate_depth.cpu();out.dense_world=backproject(s.uv,candidate_depth,s.cams,k,r,t,dist).cpu();out.dense_field_uv=field_uv.cpu();out.dense_field_cameras=field_cameras.cpu();out.dense_field_depth=field_depth.cpu();out.dense_field_world=backproject(field_uv,field_depth,field_cameras,k,r,t,dist).cpu();out.dense_history=torch::tensor(history).reshape({p.dense_iterations,3});out.diagnostics.metrics["dense_pair_supported_samples"]=static_cast<double>(s.cams.numel());out.diagnostics.metrics["dense_field_samples"]=static_cast<double>(field_cameras.numel());out.diagnostics.metrics["dense_batch_samples"]=static_cast<double>(sizes.size(0)*p.dense_samples_per_camera);out.diagnostics.metrics["dense_sparse_anchor_rmse"]=torch::sqrt(torch::mse_loss(final_sparse,sparse_z)).item<double>();
    }
    out.query_uv=p.query_uv;out.query_cameras=p.query_cameras;out.diagnostics.status=SolverStatus::CONVERGED;out.diagnostics.iterations=p.pretrain_iterations+p.dense_iterations;out.diagnostics.final_loss=loss;return out;
}
} // namespace neurodic
