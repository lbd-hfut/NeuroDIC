#include "neurodic/solver/ndef_surface_solver.hpp"

#include <ATen/Context.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <cmath>
#include <torch/cuda.h>
#include <vector>

#include "neurodic/core/exceptions.hpp"
#include "neurodic/model/ndef_depth_model.hpp"

namespace neurodic {
namespace {
torch::Tensor norm_uv(const torch::Tensor& uv, const torch::Tensor& cameras, const torch::Tensor& image_sizes) {
    // Python NDeF-DIC normalises against the complete image, not the ROI box.
    auto wh=image_sizes.index_select(0,cameras);
    return torch::stack({2.0F*uv.select(1,0)/(wh.select(1,0)-1.0F).clamp_min(1.0F)-1.0F,
                         2.0F*uv.select(1,1)/(wh.select(1,1)-1.0F).clamp_min(1.0F)-1.0F},1);
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
Samples build_samples(NDeFDepthModel& model,const NDeFSurfaceProblem& p,const torch::Tensor& mean,const torch::Tensor& sd) {
    auto cpu=torch::TensorOptions().dtype(torch::kFloat32); std::vector<float> uvs;std::vector<int64_t> cams;
    auto masks=p.roi_masks; for(int64_t c=0;c<masks.size(0);++c) for(int64_t y=0;y<masks.size(1);y+=p.dense_spacing_px) for(int64_t x=0;x<masks.size(2);x+=p.dense_spacing_px) if(masks.index({c,y,x}).item<bool>()){uvs.push_back(float(x));uvs.push_back(float(y));cams.push_back(c);}
    if(cams.empty()) throw ValidationError("NDeF dense refinement found no ROI-spaced centres");
    auto uv=torch::from_blob(uvs.data(),{static_cast<int64_t>(cams.size()),2},cpu).clone().to(p.device);auto ci=torch::from_blob(cams.data(),{static_cast<int64_t>(cams.size())},torch::TensorOptions().dtype(torch::kLong)).clone().to(p.device);
    auto k=p.intrinsics.to(p.device),r=p.rotations.to(p.device),t=p.translations.to(p.device),d=p.distortions.to(p.device),neigh=p.dense_neighbors.to(p.device),m=p.roi_masks.to(p.device);
    torch::NoGradGuard ng; auto sizes=p.image_sizes.to(p.device),depth=model.forward(norm_uv(uv,ci,sizes),ci)*sd+mean;auto world=backproject(uv,depth,ci,k,r,t,d); auto off=offsets(p.dense_patch_radius,uv.options());auto source_valid=mask_at(m,uv.unsqueeze(1)+off.unsqueeze(0),ci);auto source_inside=patch_inside_image(uv,ci,sizes,p.dense_patch_radius);
    auto targets=torch::full({uv.size(0),2},-1,torch::TensorOptions().device(p.device).dtype(torch::kLong));
    for(int slot=0;slot<2;++slot){auto tc=neigh.index_select(0,ci).select(1,slot);auto safe=tc.clamp_min(0);auto q=project(world,safe,k,r,t,d);auto center=mask_at(m,q.first.unsqueeze(1),safe).squeeze(1)&patch_inside_image(q.first,safe,sizes,p.dense_patch_radius)&(q.second>1e-8F)&(tc>=0);targets.select(1,slot).copy_(torch::where(center,tc,torch::full_like(tc,-1)));}
    auto required=static_cast<int64_t>(std::ceil(source_valid.size(1)*p.dense_min_valid_patch_ratio));auto keep=(targets>=0).any(1)&source_inside&(source_valid.sum(1)>=required);return {uv.index({keep}),ci.index({keep}),targets.index({keep}),source_valid.index({keep})};
}

struct DenseLoss { torch::Tensor photo,anchor,total; };
DenseLoss dense_loss(NDeFDepthModel& model,const NDeFSurfaceProblem& p,const Samples& s,const torch::Tensor& id,
    const torch::Tensor& sizes,const torch::Tensor& mean,const torch::Tensor& sd,const torch::Tensor& target,
    const torch::Tensor& images,const torch::Tensor& masks,const torch::Tensor& k,const torch::Tensor& r,const torch::Tensor& t,const torch::Tensor& dist,
    const torch::Tensor& sparse_uv,const torch::Tensor& sparse_c) {
    auto buv=s.uv.index_select(0,id),bc=s.cams.index_select(0,id),bt=s.targets.index_select(0,id),bvalid=s.source_patch_valid.index_select(0,id);
    auto off=offsets(p.dense_patch_radius,buv.options());auto req=static_cast<int64_t>(std::ceil(off.size(0)*p.dense_min_valid_patch_ratio));
    auto bz=model.forward(norm_uv(buv,bc,sizes),bc)*sd+mean;auto world=backproject(buv,bz,bc,k,r,t,dist);auto src_uv=buv.unsqueeze(1)+off.unsqueeze(0);auto ref=sample_bilinear(images,src_uv,bc);std::vector<torch::Tensor> edge_losses;
    for(int slot=0;slot<2;++slot){auto tc=bt.select(1,slot),present=tc>=0,safe=tc.clamp_min(0);auto q=project(world,safe,k,r,t,dist);auto tuv=q.first.unsqueeze(1)+off.unsqueeze(0);auto valid=bvalid & mask_at(masks,tuv,safe)&(q.second.unsqueeze(1)>1e-8F);auto good=present&(valid.sum(1)>=req);auto ids=torch::nonzero(good).reshape({-1});if(ids.numel()==0) continue;edge_losses.push_back(patch_znssd(ref.index_select(0,ids),sample_bilinear(images,tuv,safe).index_select(0,ids),valid.index_select(0,ids)));}
    if(edge_losses.empty()) throw ValidationError("NDeF dense refinement batch has no valid target patches");
    auto photo=torch::cat(edge_losses,0).mean();auto anchor=torch::mse_loss(model.forward(norm_uv(sparse_uv,sparse_c,sizes),sparse_c),target);return {photo,anchor,photo+p.dense_anchor_weight*anchor};
}

torch::Tensor balanced_ids(const std::vector<torch::Tensor>& pools,int64_t count) {
    std::vector<torch::Tensor> chunks;chunks.reserve(pools.size());
    for(const auto& pool:pools){auto choose=pool.numel()>=count?torch::randperm(pool.numel(),pool.options()).slice(0,0,count):torch::randint(pool.numel(),{count},pool.options());chunks.push_back(pool.index_select(0,choose));}
    return torch::cat(chunks);
}

std::pair<torch::Tensor,torch::Tensor> predict_field(NDeFDepthModel& model,const torch::Tensor& uv,const torch::Tensor& cams,
    const torch::Tensor& sizes,const torch::Tensor& mean,const torch::Tensor& sd,const torch::Tensor& k,const torch::Tensor& r,
    const torch::Tensor& t,const torch::Tensor& dist,int64_t batch_size) {
    std::vector<torch::Tensor> depths,worlds;depths.reserve((uv.size(0)+batch_size-1)/batch_size);worlds.reserve(depths.capacity());
    torch::NoGradGuard ng;
    for(int64_t start=0;start<uv.size(0);start+=batch_size){auto stop=std::min(start+batch_size,uv.size(0));auto buv=uv.slice(0,start,stop),bc=cams.slice(0,start,stop);auto depth=model.forward(norm_uv(buv,bc,sizes),bc)*sd+mean;depths.push_back(depth.cpu());worlds.push_back(backproject(buv,depth,bc,k,r,t,dist).cpu());}
    return {torch::cat(depths),torch::cat(worlds)};
}
} // namespace

NDeFSurfaceResult NDeFSurfaceSolver::solve(const NDeFSurfaceProblem& p) const {
    p.validate(); auto d=p.device; auto uv=p.sparse_uv.to(d),c=p.sparse_cameras.to(d),z=p.sparse_depth.to(d),sizes=p.image_sizes.to(d);
    // The sparse pretrain and dense phase share one model instance.  Seed both
    // generators before model construction so the dense audit has one fixed
    // C++ sparse-pretrain starting point instead of a new random field.
    torch::manual_seed(p.dense_seed);
    if (d.is_cuda()) torch::cuda::manual_seed_all(p.dense_seed);
    at::globalContext().setDeterministicAlgorithms(true, false);
    auto mean=z.mean(),sd=z.std(false).clamp_min(1e-6F),target=(z-mean)/sd; NDeFDepthModel model(sizes.size(0),p.model_options);model.to(d);
    torch::optim::AdamW opt(model.parameters(),torch::optim::AdamWOptions(p.pretrain_learning_rate).weight_decay(p.weight_decay)); double loss=0;
    for(int i=0;i<p.pretrain_iterations;++i){opt.zero_grad();auto l=torch::mse_loss(model.forward(norm_uv(uv,c,sizes),c),target);l.backward();opt.step();loss=l.item<double>();}
    NDeFSurfaceResult out; out.depth_mean=mean.item<double>();out.depth_std=sd.item<double>();
    // Preserve the sparse-SfM pretraining surface as an immutable stage-one
    // product. Dense refinement must not redefine its RMSE or visualisation.
    { torch::NoGradGuard ng; out.sparse_prediction=(model.forward(norm_uv(uv,c,sizes),c)*sd+mean).cpu();
      auto qc=p.query_cameras.to(d),qu=p.query_uv.to(d);std::vector<torch::Tensor> query_depth;
      for(int64_t start=0;start<qu.size(0);start+=p.prediction_batch_size){auto stop=std::min<int64_t>(start+p.prediction_batch_size,qu.size(0));auto buv=qu.slice(0,start,stop),bc=qc.slice(0,start,stop);query_depth.push_back((model.forward(norm_uv(buv,bc,sizes),bc)*sd+mean).cpu());}out.query_depth=torch::cat(query_depth);
      out.diagnostics.metrics["sparse_rmse"]=torch::sqrt(torch::mse_loss(out.sparse_prediction,p.sparse_depth)).item<double>(); }
    if(p.dense_iterations>0){
        auto s=build_samples(model,p,mean,sd); if(s.cams.numel()==0) throw ValidationError("NDeF dense refinement has no pair-supported samples");
        auto images=p.reference_images.to(d),masks=p.roi_masks.to(d),k=p.intrinsics.to(d),r=p.rotations.to(d),t=p.translations.to(d),dist=p.distortions.to(d),sparse_uv=p.sparse_uv.to(d),sparse_c=p.sparse_cameras.to(d),sparse_z=p.sparse_depth.to(d);
        std::vector<torch::Tensor> pools;int64_t largest=0;for(int64_t view=0;view<sizes.size(0);++view){auto pool=torch::nonzero(s.cams==view).reshape({-1});if(pool.numel()==0) throw ValidationError("NDeF dense refinement has an empty source-camera shard");largest=std::max(largest,pool.numel());pools.push_back(pool);}
        int64_t batch=std::min<int64_t>(p.dense_samples_per_camera,largest);
        if(p.dense_auto_batch){
            if(!d.is_cuda()) batch=std::min<int64_t>(p.dense_auto_batch_start,largest);
            else {
                const auto device_index=d.has_index()?d.index():c10::cuda::current_device();auto memory=c10::cuda::CUDACachingAllocator::get()->getMemoryInfo(device_index);const auto target_bytes=static_cast<size_t>(memory.first*p.dense_memory_fraction);int64_t last_good=0,probe=std::min<int64_t>(p.dense_auto_batch_start,largest),upper=largest;
                auto try_batch=[&](int64_t n){c10::cuda::CUDACachingAllocator::emptyCache();c10::cuda::CUDACachingAllocator::resetPeakStats(device_index);try{auto objective=dense_loss(model,p,s,balanced_ids(pools,n),sizes,mean,sd,target,images,masks,k,r,t,dist,sparse_uv,sparse_c);objective.total.backward();model.zero_grad();auto stats=c10::cuda::CUDACachingAllocator::getDeviceStats(device_index);return static_cast<size_t>(stats.allocated_bytes[0].peak)<=target_bytes;}catch(const c10::Error& error){model.zero_grad();c10::cuda::CUDACachingAllocator::emptyCache();if(std::string(error.what()).find("out of memory")!=std::string::npos)return false;throw;}};
                while(probe<=upper&&try_batch(probe)){last_good=probe;if(probe==upper)break;probe=std::min<int64_t>(probe*2,upper);}int64_t high=probe,low=last_good;while(high-low>1){auto mid=(low+high)/2;if(try_batch(mid)){last_good=mid;low=mid;}else high=mid;}if(last_good<1)throw ValidationError("NDeF auto batch could not fit one sample per camera");batch=last_good;
            }
        }
        torch::optim::AdamW dense_opt(model.parameters(),torch::optim::AdamWOptions(p.dense_learning_rate).weight_decay(p.weight_decay)); std::vector<float> history;torch::manual_seed(p.dense_seed);if(d.is_cuda()) torch::cuda::manual_seed_all(p.dense_seed);
        const int64_t steps_per_epoch=(largest+batch-1)/batch;
        for(int epoch=0;epoch<p.dense_epochs;++epoch){std::vector<torch::Tensor> order;std::vector<int64_t> position(pools.size(),0);for(const auto& pool:pools)order.push_back(pool.index_select(0,torch::randperm(pool.numel(),pool.options())));
            for(int64_t step=0;step<steps_per_epoch;++step){std::vector<torch::Tensor> chunks;for(size_t view=0;view<pools.size();++view){std::vector<torch::Tensor> pieces;int64_t remaining=batch;while(remaining>0){auto available=order[view].numel()-position[view];auto take=std::min(remaining,available);pieces.push_back(order[view].slice(0,position[view],position[view]+take));position[view]+=take;remaining-=take;if(position[view]>=order[view].numel()){order[view]=pools[view].index_select(0,torch::randperm(pools[view].numel(),pools[view].options()));position[view]=0;}}chunks.push_back(torch::cat(pieces));}auto id=torch::cat(chunks);dense_opt.zero_grad();auto objective=dense_loss(model,p,s,id,sizes,mean,sd,target,images,masks,k,r,t,dist,sparse_uv,sparse_c);objective.total.backward();dense_opt.step();history.push_back(objective.photo.detach().item<float>());history.push_back(objective.anchor.detach().item<float>());history.push_back(objective.total.detach().item<float>());loss=objective.total.item<double>();}}
        auto candidate=predict_field(model,s.uv,s.cams,sizes,mean,sd,k,r,t,dist,p.prediction_batch_size);auto field_uv=p.query_uv.to(d),field_cameras=p.query_cameras.to(d);auto field=predict_field(model,field_uv,field_cameras,sizes,mean,sd,k,r,t,dist,p.prediction_batch_size);torch::NoGradGuard ng;auto final_sparse=model.forward(norm_uv(sparse_uv,sparse_c,sizes),sparse_c)*sd+mean;out.dense_uv=s.uv.cpu();out.dense_cameras=s.cams.cpu();out.dense_targets=s.targets.cpu();out.dense_depth=candidate.first;out.dense_world=candidate.second;out.dense_field_uv=p.query_uv;out.dense_field_cameras=p.query_cameras;out.dense_field_depth=field.first;out.dense_field_world=field.second;out.dense_history=torch::tensor(history).reshape({-1,3});out.diagnostics.metrics["dense_pair_supported_samples"]=static_cast<double>(s.cams.numel());out.diagnostics.metrics["dense_field_samples"]=static_cast<double>(field_cameras.numel());out.diagnostics.metrics["dense_batch_per_camera"]=static_cast<double>(batch);out.diagnostics.metrics["dense_steps_per_epoch"]=static_cast<double>(steps_per_epoch);out.diagnostics.metrics["dense_sparse_anchor_rmse"]=torch::sqrt(torch::mse_loss(final_sparse,sparse_z)).item<double>();
    }
    out.query_uv=p.query_uv;out.query_cameras=p.query_cameras;out.diagnostics.status=SolverStatus::CONVERGED;out.diagnostics.iterations=p.pretrain_iterations+(out.dense_history.defined()?out.dense_history.size(0):0);out.diagnostics.final_loss=loss;return out;
}
} // namespace neurodic
