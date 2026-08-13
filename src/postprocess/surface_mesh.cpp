#include "neurodic/postprocess/surface_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <vector>

#include "neurodic/core/exceptions.hpp"
#include "neurodic/postprocess/filtering.hpp"

namespace neurodic {
namespace {
using Vec = std::array<double, 3>;
double dot(const Vec& a, const Vec& b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
Vec sub(const Vec& a, const Vec& b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
Vec cross(const Vec& a, const Vec& b) { return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]}; }
double norm(const Vec& a) { return std::sqrt(dot(a, a)); }
Vec scale(const Vec& a, double value) { return {a[0]*value, a[1]*value, a[2]*value}; }
Vec normalized(const Vec& a) { const double n = norm(a); return n > 1e-15 ? scale(a, 1.0/n) : Vec{0.,0.,1.}; }

Vec normal_from_neighbors(const std::vector<Vec>& points, const torch::TensorAccessor<int64_t, 2>& knn,
                          int64_t point, int64_t k) {
    Vec center{0.,0.,0.};
    for (int64_t j = 0; j < k; ++j) { const auto& p = points[static_cast<size_t>(knn[point][j])]; for (int d=0;d<3;++d) center[d]+=p[d]; }
    for (double& value : center) value /= static_cast<double>(k);
    std::array<std::array<double,3>,3> c{};
    for (int64_t j = 0; j < k; ++j) { const Vec v = sub(points[static_cast<size_t>(knn[point][j])], center);
        for (int a=0;a<3;++a) for(int b=0;b<3;++b) c[a][b] += v[a]*v[b]; }
    // Symmetric 3x3 Jacobi eigensolver; the smallest eigenvector is the normal.
    std::array<std::array<double,3>,3> q{{{{1.,0.,0.}},{{0.,1.,0.}},{{0.,0.,1.}}}};
    for (int it=0;it<16;++it) { int a=0,b=1; if(std::abs(c[0][2])>std::abs(c[a][b])){a=0;b=2;} if(std::abs(c[1][2])>std::abs(c[a][b])){a=1;b=2;} if(std::abs(c[a][b])<1e-15) break;
        const double t=.5*std::atan2(2*c[a][b],c[b][b]-c[a][a]), cs=std::cos(t), sn=std::sin(t), aa=c[a][a], bb=c[b][b], ab=c[a][b];
        for(int r=0;r<3;++r)if(r!=a&&r!=b){const double ra=c[r][a],rb=c[r][b];c[r][a]=c[a][r]=cs*ra-sn*rb;c[r][b]=c[b][r]=sn*ra+cs*rb;}
        c[a][a]=cs*cs*aa-2*sn*cs*ab+sn*sn*bb;c[b][b]=sn*sn*aa+2*sn*cs*ab+cs*cs*bb;c[a][b]=c[b][a]=0.;
        for(int r=0;r<3;++r){const double ra=q[r][a],rb=q[r][b];q[r][a]=cs*ra-sn*rb;q[r][b]=sn*ra+cs*rb;}}
    int id=c[1][1]<c[0][0]?1:0; if(c[2][2]<c[id][id]) id=2; return normalized({q[0][id],q[1][id],q[2][id]});
}

double median(std::vector<double> values) { const auto m=values.size()/2; std::nth_element(values.begin(),values.begin()+static_cast<std::ptrdiff_t>(m),values.end()); return values[m]; }
}  // namespace

SurfaceMesh triangulate_pin_multi_surface(const torch::Tensor& input, SurfaceMeshOptions options) {
    if (!input.defined() || input.dim()!=2 || input.size(1)!=3) throw ValidationError("Surface meshing expects finite [N,3] points");
    if (options.k_neighbors<3 || options.min_triangle_quality<0.0 || options.min_triangle_quality>1.0) throw ValidationError("Invalid surface meshing options");
    auto vertices=input.detach().to(torch::kCPU).to(torch::kFloat64).contiguous(); const auto n=vertices.size(0);
    SurfaceMesh result; result.vertices=vertices; result.faces=torch::empty({0,3},torch::kLong); result.normals=torch::empty({n,3},torch::kFloat64); result.quality=torch::empty({0},torch::kFloat64);
    if(n<=options.k_neighbors) return result;
    const auto k=std::min<int64_t>(options.k_neighbors,n-1); auto knn=knn_indices_3d(vertices,k); auto ids=knn.accessor<int64_t,2>(); auto v=vertices.accessor<double,2>();
    std::vector<Vec> points(static_cast<size_t>(n)); for(int64_t i=0;i<n;++i) points[static_cast<size_t>(i)]={v[i][0],v[i][1],v[i][2]};
    std::vector<Vec> normals(static_cast<size_t>(n)); std::vector<double> spacing; spacing.reserve(static_cast<size_t>(n));
    for(int64_t i=0;i<n;++i){normals[static_cast<size_t>(i)]=normal_from_neighbors(points,ids,i,k); std::vector<double>d;for(int64_t j=0;j<k;++j)d.push_back(norm(sub(points[static_cast<size_t>(ids[i][j])],points[static_cast<size_t>(i)])));spacing.push_back(median(std::move(d)));}
    result.median_spacing=median(spacing); result.max_edge_length=options.max_edge_length>0.0?options.max_edge_length:2.5*result.median_spacing;
    auto normal_tensor=torch::empty({n,3},torch::kFloat64); auto na=normal_tensor.accessor<double,2>(); for(int64_t i=0;i<n;++i)for(int d=0;d<3;++d)na[i][d]=normals[static_cast<size_t>(i)][d]; result.normals=normal_tensor;
    std::set<std::array<int64_t,3>> unique; std::vector<std::array<int64_t,3>> faces; std::vector<double> qualities;
    const auto topology_neighbors=std::min<int64_t>(k,6); // A manifold vertex has about six Delaunay neighbours.
    for(int64_t center=0;center<n;++center){ const auto& p=points[static_cast<size_t>(center)]; const auto& normal=normals[static_cast<size_t>(center)]; Vec tangent{1.,0.,0.}; if(std::abs(dot(tangent,normal))>.9)tangent={0.,1.,0.}; tangent=normalized(sub(tangent,scale(normal,dot(tangent,normal)))); const Vec bitangent=cross(normal,tangent);
        std::vector<std::pair<double,int64_t>> ring; ring.reserve(static_cast<size_t>(topology_neighbors)); for(int64_t j=0;j<topology_neighbors;++j){const auto id=ids[center][j];const Vec d=sub(points[static_cast<size_t>(id)],p);ring.emplace_back(std::atan2(dot(d,bitangent),dot(d,tangent)),id);} std::sort(ring.begin(),ring.end());
        for(size_t j=0;j<ring.size();++j){int64_t a=ring[j].second,b=ring[(j+1)%ring.size()].second;
            // A tangent-plane fan alone creates overlapping triangles.  Keep
            // only edges independently supported by the k-NN graph.
            bool mutual=false; for(int64_t q=0;q<k;++q) if(ids[a][q]==b){mutual=true;break;}
            if(!mutual || dot(normal,normals[static_cast<size_t>(a)])<0.707 || dot(normal,normals[static_cast<size_t>(b)])<0.707)continue;
            const Vec pa=sub(points[static_cast<size_t>(a)],p),pb=sub(points[static_cast<size_t>(b)],p),ab=sub(points[static_cast<size_t>(b)],points[static_cast<size_t>(a)]);const double la=norm(pa),lb=norm(pb),lc=norm(ab);if(std::max({la,lb,lc})>result.max_edge_length)continue;const double area2=norm(cross(pa,pb));const double quality=2.0*std::sqrt(3.0)*area2/(la*la+lb*lb+lc*lc);if(quality<options.min_triangle_quality)continue;std::array<int64_t,3> key{center,a,b};std::sort(key.begin(),key.end());if(!unique.insert(key).second)continue;auto face=std::array<int64_t,3>{center,a,b};if(dot(cross(pa,pb),normal)<0.0)std::swap(face[1],face[2]);faces.push_back(face);qualities.push_back(quality);}}
    if(!faces.empty()){result.faces=torch::from_blob(faces.data(),{static_cast<int64_t>(faces.size()),3},torch::kLong).clone();result.quality=torch::from_blob(qualities.data(),{static_cast<int64_t>(qualities.size())},torch::kFloat64).clone();}
    return result;
}
}  // namespace neurodic
