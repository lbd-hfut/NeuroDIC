#include "neurodic/postprocess/filtering.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>

#include "neurodic/core/exceptions.hpp"

namespace neurodic {
namespace {
using Point = std::array<double, 3>;

struct Node { int point{-1}; int axis{0}; int left{-1}; int right{-1}; };

class KDTree {
public:
    explicit KDTree(std::vector<Point> points) : points_(std::move(points)), ids_(points_.size()) {
        std::iota(ids_.begin(), ids_.end(), 0); nodes_.reserve(points_.size());
        root_ = build(0, static_cast<int>(ids_.size()), 0);
    }
    std::vector<int> nearest(const Point& query, int k) const {
        using Item = std::pair<double, int>;
        std::priority_queue<Item> heap;
        search(root_, query, k, heap);
        std::vector<int> result; result.reserve(heap.size());
        while (!heap.empty()) { result.push_back(heap.top().second); heap.pop(); }
        return result;
    }
private:
    int build(int first, int last, int depth) {
        if (first >= last) return -1;
        const int axis = depth % 3, middle = first + (last - first) / 2;
        std::nth_element(ids_.begin() + first, ids_.begin() + middle, ids_.begin() + last,
                         [&](int a, int b) { return points_[a][axis] < points_[b][axis]; });
        const int node = static_cast<int>(nodes_.size()); nodes_.push_back({ids_[middle], axis, -1, -1});
        nodes_[node].left = build(first, middle, depth + 1);
        nodes_[node].right = build(middle + 1, last, depth + 1);
        return node;
    }
    void search(int node_id, const Point& query, int k,
                std::priority_queue<std::pair<double, int>>& heap) const {
        if (node_id < 0) return;
        const auto& node = nodes_[node_id]; const auto& point = points_[node.point];
        double d2 = 0.0; for (int d = 0; d < 3; ++d) d2 += (point[d] - query[d]) * (point[d] - query[d]);
        if (static_cast<int>(heap.size()) < k) heap.emplace(d2, node.point);
        else if (d2 < heap.top().first) { heap.pop(); heap.emplace(d2, node.point); }
        const double delta = query[node.axis] - point[node.axis];
        const int near = delta <= 0.0 ? node.left : node.right;
        const int far = delta <= 0.0 ? node.right : node.left;
        search(near, query, k, heap);
        if (static_cast<int>(heap.size()) < k || delta * delta < heap.top().first) search(far, query, k, heap);
    }
    std::vector<Point> points_; std::vector<int> ids_; std::vector<Node> nodes_; int root_{-1};
};

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
    double value = values[middle];
    if (values.size() % 2 == 0) {
        std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle - 1), values.end());
        value = 0.5 * (value + values[middle - 1]);
    }
    return value;
}

Point smallest_eigenvector(std::array<std::array<double, 3>, 3> matrix) {
    std::array<std::array<double, 3>, 3> vectors{{{{1.,0.,0.}}, {{0.,1.,0.}}, {{0.,0.,1.}}}};
    for (int iteration = 0; iteration < 16; ++iteration) {
        int p = 0, q = 1;
        if (std::abs(matrix[0][2]) > std::abs(matrix[p][q])) { p = 0; q = 2; }
        if (std::abs(matrix[1][2]) > std::abs(matrix[p][q])) { p = 1; q = 2; }
        if (std::abs(matrix[p][q]) < 1e-15) break;
        const double theta = 0.5 * std::atan2(2.0 * matrix[p][q], matrix[q][q] - matrix[p][p]);
        const double c = std::cos(theta), s = std::sin(theta);
        for (int r = 0; r < 3; ++r) if (r != p && r != q) {
            const double rp = matrix[r][p], rq = matrix[r][q];
            matrix[r][p] = matrix[p][r] = c * rp - s * rq;
            matrix[r][q] = matrix[q][r] = s * rp + c * rq;
        }
        const double pp = matrix[p][p], qq = matrix[q][q], pq = matrix[p][q];
        matrix[p][p] = c*c*pp - 2*s*c*pq + s*s*qq;
        matrix[q][q] = s*s*pp + 2*s*c*pq + c*c*qq;
        matrix[p][q] = matrix[q][p] = 0.0;
        for (int r = 0; r < 3; ++r) { const double vp = vectors[r][p], vq = vectors[r][q];
            vectors[r][p] = c * vp - s * vq; vectors[r][q] = s * vp + c * vq; }
    }
    int smallest = 0; if (matrix[1][1] < matrix[smallest][smallest]) smallest = 1;
    if (matrix[2][2] < matrix[smallest][smallest]) smallest = 2;
    return {vectors[0][smallest], vectors[1][smallest], vectors[2][smallest]};
}

double robust_threshold(double center, double mad, double factor) {
    return mad > std::numeric_limits<double>::epsilon() ? center + factor * mad
        : std::max(center * (1.0 + 0.25 * factor), center + 1e-6);
}

}  // namespace

SurfaceCleanupResult clean_pin_multi_surface(const torch::Tensor& input, int64_t k_neighbors, double mad_factor) {
    if (!input.defined() || input.dim() != 2 || input.size(1) != 3)
        throw ValidationError("PIN multi surface cleanup expects points [N,3]");
    if (k_neighbors < 1 || mad_factor < 0.0) throw ValidationError("Invalid surface cleanup options");
    auto points_cpu = input.detach().to(torch::kCPU).to(torch::kFloat64).contiguous();
    const auto count = points_cpu.size(0);
    SurfaceCleanupResult result;
    result.inlier_mask = torch::ones({count}, torch::kBool);
    result.neighbor_distance = torch::zeros({count}, torch::kFloat64);
    result.plane_residual = torch::zeros({count}, torch::kFloat64);
    if (count == 0 || mad_factor == 0.0 || count <= k_neighbors) return result;
    const auto accessor = points_cpu.accessor<double, 2>();
    std::vector<Point> points(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
        points[static_cast<size_t>(i)] = {accessor[i][0], accessor[i][1], accessor[i][2]};
        if (!std::isfinite(accessor[i][0]) || !std::isfinite(accessor[i][1]) || !std::isfinite(accessor[i][2]))
            throw ValidationError("PIN multi surface cleanup requires finite points");
    }
    KDTree tree(points); const int k = static_cast<int>(std::min<int64_t>(k_neighbors, count - 1));
    std::vector<double> density(static_cast<size_t>(count)), residual(static_cast<size_t>(count));
    for (int64_t index = 0; index < count; ++index) {
        auto neighbours = tree.nearest(points[static_cast<size_t>(index)], k + 1);
        neighbours.erase(std::remove(neighbours.begin(), neighbours.end(), static_cast<int>(index)), neighbours.end());
        if (static_cast<int>(neighbours.size()) > k) neighbours.resize(static_cast<size_t>(k));
        std::vector<double> distances; distances.reserve(neighbours.size()); Point center{0., 0., 0.};
        for (int id : neighbours) { const auto& p = points[static_cast<size_t>(id)];
            for (int d = 0; d < 3; ++d) center[d] += p[d];
            const auto& q = points[static_cast<size_t>(index)]; double d2 = 0.; for (int d = 0; d < 3; ++d) d2 += (p[d]-q[d])*(p[d]-q[d]); distances.push_back(std::sqrt(d2)); }
        for (double& value : center) value /= static_cast<double>(neighbours.size());
        density[static_cast<size_t>(index)] = median(std::move(distances));
        std::array<std::array<double, 3>, 3> covariance{};
        for (int id : neighbours) { const auto& p = points[static_cast<size_t>(id)]; Point v{p[0]-center[0], p[1]-center[1], p[2]-center[2]};
            for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) covariance[a][b] += v[a]*v[b]; }
        const auto normal = smallest_eigenvector(covariance); const auto& q = points[static_cast<size_t>(index)];
        residual[static_cast<size_t>(index)] = std::abs((q[0]-center[0])*normal[0] + (q[1]-center[1])*normal[1] + (q[2]-center[2])*normal[2]);
    }
    result.neighbor_distance = torch::from_blob(density.data(), {count}, torch::kFloat64).clone();
    result.plane_residual = torch::from_blob(residual.data(), {count}, torch::kFloat64).clone();
    result.neighbor_distance_median = median(density); std::vector<double> deviations(density.size());
    for (size_t i = 0; i < density.size(); ++i) deviations[i] = std::abs(density[i] - result.neighbor_distance_median);
    result.neighbor_distance_mad = median(deviations);
    result.neighbor_distance_threshold = robust_threshold(result.neighbor_distance_median, result.neighbor_distance_mad, mad_factor);
    result.plane_residual_median = median(residual);
    for (size_t i = 0; i < residual.size(); ++i) deviations[i] = std::abs(residual[i] - result.plane_residual_median);
    result.plane_residual_mad = median(deviations);
    result.plane_residual_threshold = robust_threshold(result.plane_residual_median, result.plane_residual_mad, mad_factor);
    result.inlier_mask = (result.neighbor_distance <= result.neighbor_distance_threshold) &
                         (result.plane_residual <= result.plane_residual_threshold);
    return result;
}

torch::Tensor knn_indices_3d(const torch::Tensor& input, int64_t k_neighbors) {
    if (!input.defined() || input.dim() != 2 || input.size(1) != 3)
        throw ValidationError("3D k-NN expects finite points [N,3]");
    const auto count = input.size(0);
    if (k_neighbors < 1 || count <= k_neighbors) throw ValidationError("3D k-NN needs more points than neighbours");
    auto points_cpu = input.detach().to(torch::kCPU).to(torch::kFloat64).contiguous();
    const auto a = points_cpu.accessor<double, 2>(); std::vector<Point> points(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) { points[static_cast<size_t>(i)] = {a[i][0], a[i][1], a[i][2]};
        if (!std::isfinite(a[i][0]) || !std::isfinite(a[i][1]) || !std::isfinite(a[i][2])) throw ValidationError("3D k-NN requires finite points"); }
    KDTree tree(points); auto result = torch::empty({count, k_neighbors}, torch::kLong); auto out = result.accessor<int64_t, 2>();
    for (int64_t i = 0; i < count; ++i) { auto ids = tree.nearest(points[static_cast<size_t>(i)], static_cast<int>(k_neighbors + 1));
        ids.erase(std::remove(ids.begin(), ids.end(), static_cast<int>(i)), ids.end());
        if (static_cast<int64_t>(ids.size()) < k_neighbors) throw ValidationError("3D k-NN query returned too few neighbours");
        for (int64_t j = 0; j < k_neighbors; ++j) out[i][j] = ids[static_cast<size_t>(j)]; }
    return result;
}

MeshCleanupResult clean_pin_multi_mesh(const torch::Tensor& vertices_input, const torch::Tensor& faces_input,
                                       const torch::Tensor& quality_input, double overlap_distance,
                                       double min_triangle_quality) {
    if (!vertices_input.defined() || vertices_input.dim() != 2 || vertices_input.size(1) != 3 ||
        !faces_input.defined() || faces_input.dim() != 2 || faces_input.size(1) != 3)
        throw ValidationError("Mesh cleanup expects vertices [N,3] and faces [M,3]");
    if (min_triangle_quality < 0.0 || min_triangle_quality > 1.0) throw ValidationError("Invalid triangle quality threshold");
    auto vertices = vertices_input.detach().to(torch::kCPU).to(torch::kFloat64).contiguous();
    auto faces = faces_input.detach().to(torch::kCPU).to(torch::kLong).contiguous();
    const auto count = faces.size(0), vertices_count = vertices.size(0);
    MeshCleanupResult result; result.face_mask = torch::zeros({count}, torch::kBool);
    result.face_quality = torch::zeros({count}, torch::kFloat64); if (count == 0) return result;
    if (quality_input.defined() && quality_input.numel() != count) throw ValidationError("Mesh quality must be [M]");
    auto supplied = quality_input.defined() ? quality_input.detach().to(torch::kCPU).to(torch::kFloat64).reshape({count}) : torch::Tensor();
    const auto v = vertices.accessor<double, 2>(); const auto f = faces.accessor<int64_t, 2>();
    struct Face { int64_t id; std::array<int64_t,3> ids; Point centroid, normal; double quality, edge; };
    std::vector<Face> candidates; candidates.reserve(static_cast<size_t>(count)); std::vector<double> edges;
    for (int64_t row=0; row<count; ++row) {
        std::array<int64_t,3> ids{f[row][0],f[row][1],f[row][2]};
        if (ids[0]<0 || ids[1]<0 || ids[2]<0 || ids[0]>=vertices_count || ids[1]>=vertices_count || ids[2]>=vertices_count || ids[0]==ids[1] || ids[1]==ids[2] || ids[0]==ids[2]) continue;
        Point p[3]{{v[ids[0]][0],v[ids[0]][1],v[ids[0]][2]},
                   {v[ids[1]][0],v[ids[1]][1],v[ids[1]][2]},
                   {v[ids[2]][0],v[ids[2]][1],v[ids[2]][2]}};
        Point a{p[1][0]-p[0][0],p[1][1]-p[0][1],p[1][2]-p[0][2]}, b{p[2][0]-p[0][0],p[2][1]-p[0][1],p[2][2]-p[0][2]};
        Point cross{a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; const double area2=std::sqrt(cross[0]*cross[0]+cross[1]*cross[1]+cross[2]*cross[2]);
        double length[3]{}; for(int e=0;e<3;++e){const auto& x=p[e];const auto& y=p[(e+1)%3];length[e]=std::sqrt((x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1])+(x[2]-y[2])*(x[2]-y[2]));}
        const double computed=2.0*std::sqrt(3.0)*area2/(length[0]*length[0]+length[1]*length[1]+length[2]*length[2]);
        const double q=supplied.defined()?supplied[row].item<double>():computed; if (!std::isfinite(q) || computed<min_triangle_quality || q<min_triangle_quality) continue;
        Point centroid{(p[0][0]+p[1][0]+p[2][0])/3.,(p[0][1]+p[1][1]+p[2][1])/3.,(p[0][2]+p[1][2]+p[2][2])/3.}; for(double& x:cross)x/=area2;
        candidates.push_back({row,ids,centroid,cross,q,(length[0]+length[1]+length[2])/3.}); edges.push_back((length[0]+length[1]+length[2])/3.);
    }
    if (candidates.empty()) return result;
    result.mean_edge_length=std::accumulate(edges.begin(),edges.end(),0.0)/static_cast<double>(edges.size());
    // MultiDIC uses ~0.6 edge length after ray/face intersection.  Here we
    // compare centroids only, so 0.2 is deliberately conservative and avoids
    // deleting adjacent, non-overlapping triangles.
    result.overlap_distance=overlap_distance>0.0?overlap_distance:0.2*result.mean_edge_length;
    std::sort(candidates.begin(),candidates.end(),[](const Face&a,const Face&b){return a.quality>b.quality;});
    struct Cell { int x,y,z; bool operator==(const Cell& o)const{return x==o.x&&y==o.y&&z==o.z;} };
    struct Hash { size_t operator()(const Cell& c)const{return static_cast<size_t>(c.x)*73856093U^static_cast<size_t>(c.y)*19349663U^static_cast<size_t>(c.z)*83492791U;} };
    std::unordered_map<Cell,std::vector<size_t>,Hash> grid; std::set<std::array<int64_t,3>> triples; std::vector<Face> kept;
    for (const auto& face : candidates) {
        auto key=face.ids;std::sort(key.begin(),key.end()); if(!triples.insert(key).second)continue;
        Cell cell{static_cast<int>(std::floor(face.centroid[0]/result.overlap_distance)),static_cast<int>(std::floor(face.centroid[1]/result.overlap_distance)),static_cast<int>(std::floor(face.centroid[2]/result.overlap_distance))}; bool overlaps=false;
        for(int dx=-1;dx<=1&&!overlaps;++dx)for(int dy=-1;dy<=1&&!overlaps;++dy)for(int dz=-1;dz<=1&&!overlaps;++dz){auto it=grid.find({cell.x+dx,cell.y+dy,cell.z+dz});if(it==grid.end())continue;for(size_t id:it->second){const auto& prior=kept[id];int shared=0;for(auto a:face.ids)for(auto b:prior.ids)if(a==b)++shared;if(shared>0)continue;const double nx=face.normal[0]*prior.normal[0]+face.normal[1]*prior.normal[1]+face.normal[2]*prior.normal[2];const double d=std::sqrt((face.centroid[0]-prior.centroid[0])*(face.centroid[0]-prior.centroid[0])+(face.centroid[1]-prior.centroid[1])*(face.centroid[1]-prior.centroid[1])+(face.centroid[2]-prior.centroid[2])*(face.centroid[2]-prior.centroid[2]));if(std::abs(nx)>0.95&&d<result.overlap_distance){overlaps=true;break;}}}
        if(overlaps)continue; result.face_mask[face.id]=true;result.face_quality[face.id]=face.quality;grid[cell].push_back(kept.size());kept.push_back(face);
    }
    return result;
}

LocalDisplacementConsistencyResult compute_local_displacement_consistency(
    const torch::Tensor& coordinates_input, const torch::Tensor& displacement_input,
    const torch::Tensor& valid_input, int64_t k_neighbors, double mad_factor) {
    if (!coordinates_input.defined() || coordinates_input.dim()!=2 || coordinates_input.size(1)!=3 ||
        !displacement_input.defined() || displacement_input.sizes()!=coordinates_input.sizes())
        throw ValidationError("Local displacement consistency expects matching [N,3] fields");
    if (k_neighbors < 3 || mad_factor < 0.0) throw ValidationError("Invalid local consistency options");
    auto x=coordinates_input.detach().to(torch::kCPU).to(torch::kFloat64).contiguous();
    auto u=displacement_input.detach().to(torch::kCPU).to(torch::kFloat64).contiguous(); const auto n=x.size(0);
    auto valid=valid_input.defined()?valid_input.detach().to(torch::kCPU).to(torch::kBool).reshape({-1}):torch::isfinite(x).all(1)&torch::isfinite(u).all(1);
    if(valid.numel()!=n) throw ValidationError("Local displacement consistency valid must be [N]");
    valid=valid&torch::isfinite(x).all(1)&torch::isfinite(u).all(1);
    LocalDisplacementConsistencyResult result;
    result.predicted_displacement=torch::full({n,3},std::numeric_limits<double>::quiet_NaN(),torch::kFloat64);
    result.residual=torch::full({n},std::numeric_limits<double>::quiet_NaN(),torch::kFloat64);
    result.inlier_mask=torch::zeros({n},torch::kBool);
    auto ids=torch::nonzero(valid).reshape({-1}); if(ids.numel()<=k_neighbors) return result;
    auto px=x.index_select(0,ids), pu=u.index_select(0,ids); const auto k=std::min<int64_t>(k_neighbors,ids.numel()-1);
    auto nearest=knn_indices_3d(px,k);
    auto xa=px.accessor<double,2>();
    auto ua=pu.accessor<double,2>();
    auto ni=nearest.accessor<int64_t,2>();
    auto predicted=result.predicted_displacement.accessor<double,2>(); auto residual=result.residual.accessor<double,1>();
    for(int64_t row=0;row<ids.numel();++row) {
        std::array<double,3> cx{},cu{}; for(int64_t j=0;j<k;++j){const auto q=ni[row][j];for(int d=0;d<3;++d){cx[d]+=xa[q][d];cu[d]+=ua[q][d];}}
        for(int d=0;d<3;++d){cx[d]/=k;cu[d]/=k;}
        auto design=torch::empty({k,3},torch::kFloat64), values=torch::empty({k,3},torch::kFloat64); auto da=design.accessor<double,2>(), va=values.accessor<double,2>();
        for(int64_t j=0;j<k;++j){const auto q=ni[row][j];for(int d=0;d<3;++d){da[j][d]=xa[q][d]-cx[d];va[j][d]=ua[q][d]-cu[d];}}
        auto a=std::get<0>(torch::linalg_lstsq(design,values)); if(!torch::isfinite(a).all().item<bool>())continue; auto aa=a.accessor<double,2>();
        const auto original=ids[row].item<int64_t>(); double r2=0.; for(int out=0;out<3;++out){double value=cu[out];for(int in=0;in<3;++in)value+=(xa[row][in]-cx[in])*aa[in][out];predicted[original][out]=value;const double e=ua[row][out]-value;r2+=e*e;} residual[original]=std::sqrt(r2);
    }
    auto usable=torch::isfinite(result.residual); auto values=result.residual.index({usable}).contiguous(); if(values.numel()==0)return result;
    std::vector<double> raw(values.data_ptr<double>(),values.data_ptr<double>()+values.numel()); result.residual_median=median(raw);for(double& value:raw)value=std::abs(value-result.residual_median);result.residual_mad=median(raw);result.residual_threshold=robust_threshold(result.residual_median,result.residual_mad,mad_factor);
    result.inlier_mask=valid&torch::isfinite(result.residual)&(result.residual<=result.residual_threshold); return result;
}

}  // namespace neurodic
