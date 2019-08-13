#include <se/octree.hpp>
#include <se/io/ply_io.hpp>
#include "../src/multires_bfusion/alloc_impl.hpp"
#include <se/volume_traits.hpp>
#include <gtest/gtest.h>
#include <chrono>
#include <ctime>

// Returned distance when ray doesn't intersect sphere
#define SENSOR_LIMIT 5

// Number of frames to move from start to end position
#define FRAMES 5

// Activate (1) and deactivate (0) depth dependent noise
#define NOISE 0

// Box intersection
#define RIGHT	0
#define LEFT	1
#define MIDDLE	2

struct camera_parameter {
public:
  camera_parameter() {};

  camera_parameter(float FOV, Eigen::Vector2i image_size, Eigen::Matrix4f Twc)
      : image_size_(image_size), Twc_(Twc) {
    focal_length_pix_ = image_size_.x()/2*std::tan(M_PI*FOV/360);
    K_ << focal_length_pix_, 0                , image_size.x()/2, 0,
        0                , focal_length_pix_, image_size.y()/2, 0,
        0                , 0                , 1               , 0,
        0                , 0                , 0               , 1;
  };
  float focalLengthPix() {return focal_length_pix_;}
  void setPose(Eigen::Matrix4f Twc) {Twc_ = Twc;}
  Eigen::Vector2i imageSize() {return image_size_;}
  Eigen::Matrix4f Twc() {return Twc_;}
  Eigen::Vector3f twc() {return Twc_.topRightCorner(3,1);}
  Eigen::Matrix3f Rwc() {return Twc_.topLeftCorner(3,3);};
  Eigen::Matrix4f K() {return K_;}

private:
  float focal_length_pix_; //[vox]
  Eigen::Vector2i image_size_;
  Eigen::Matrix4f Twc_;
  Eigen::Matrix4f K_;
};

struct ray {
public:
  ray(camera_parameter& camera_parameter)
      : direction_(Eigen::Vector3f(-1.f,-1.f,-1.f)),
        origin_(camera_parameter.twc()),
        t_wc_(camera_parameter.twc()) {
    Eigen::Matrix4f inv_K = (camera_parameter.K()).inverse();
    Eigen::Matrix4f inv_Tcw = camera_parameter.Twc();

    inv_P_ = inv_Tcw * inv_K;
  };

  void operator()(int pixel_u, int pixel_v, float depth = 1) {
    // Update world vortex
    vertex_w_ = (inv_P_ * Eigen::Vector3f(depth*(pixel_u + 0.5f), depth*(pixel_v + 0.5f), depth).homogeneous()).head<3>();

    // Update direction
    direction_ = vertex_w_ - t_wc_;
    range_ = direction_.norm();
    direction_.normalize();
  }

  Eigen::Vector3f origin() {return  origin_;}
  Eigen::Vector3f direction() {return direction_;}
  Eigen::Vector3f vertex_w() {return vertex_w_;}
  float range() {return range_;}

private:
  Eigen::Vector3f direction_;
  Eigen::Vector3f origin_;
  Eigen::Vector3f vertex_w_;
  Eigen::Vector3f t_wc_;
  Eigen::Matrix4f inv_P_;
  float range_;
};

struct obstacle {
  virtual float intersect(ray& ray) = 0;
};

struct sphere_obstacle : obstacle {
public:
  sphere_obstacle() {};
  sphere_obstacle(Eigen::Vector3f center, float radius)
      : center_(center), radius_(radius) {};

  sphere_obstacle(camera_parameter camera_parameter, Eigen::Vector2f center_angle,
                  float center_distance, float radius)
      : radius_(radius) {
    Eigen::Matrix3f Rwc = camera_parameter.Rwc();
    Eigen::Vector3f twc = camera_parameter.twc();
    float dist_y = std::sin(center_angle.x())*center_distance;
    float dist_x = std::cos(center_angle.x())*std::sin(center_angle.y())*center_distance;
    float dist_z = std::cos(center_angle.x())*std::cos(center_angle.y())*center_distance;
    Eigen::Vector3f dist(dist_x, dist_y, dist_z);
    center_ = Rwc*dist + twc;
  };

  float intersect(ray& ray) {
    float dist(0);
    Eigen::Vector3f oc = ray.origin() - center_;
    float a = ray.direction().dot(ray.direction());
    float b = 2.0 * oc.dot(ray.direction());
    float c = oc.dot(oc) - radius_*radius_;
    float discriminant = b*b - 4*a*c;
    if (discriminant >= 0) {
      float dist_tmp = (-b - sqrt(discriminant))/(2.0*a);
      if (dist_tmp > 0 && dist_tmp < SENSOR_LIMIT)
        dist = dist_tmp;
    }
    return dist;
  };

  Eigen::Vector3f center() {return center_;}
  float radius() {return radius_;}

private:
  Eigen::Vector3f center_;
  float radius_;
};

struct box_obstacle : obstacle {
public:
  box_obstacle() {};
  box_obstacle(Eigen::Vector3f center, float depth, float width, float height)
      : center_(center), dim_(Eigen::Vector3f(depth, width, height)) {
    min_corner_ = center - Eigen::Vector3f(depth, width, height);
    max_corner_ = center + Eigen::Vector3f(depth, width, height);
  };

  box_obstacle(Eigen::Vector3f center, Eigen::Vector3f dim)
      : center_(center), dim_(dim) {
    min_corner_ = center - dim/2;
    max_corner_ = center + dim/2;
  };

  float intersect(ray& ray) {
    float dist(0);
    /*
    Fast Ray-Box Intersection
    by Andrew Woo
    from "Graphics Gems", Academic Press, 1990
    */
    int num_dim = 3;
    Eigen::Vector3f hit_point = -1*Eigen::Vector3f::Ones();				/* hit point */
    {
      bool inside = true;
      Eigen::Vector3i quadrant;
      int which_plane;
      Eigen::Vector3f max_T;
      Eigen::Vector3f candidate_plane;

      /* Find candidate planes; this loop can be avoided if
         rays cast all from the eye(assume perpsective view) */
      for (int i = 0; i < num_dim; i++)
        if(ray.origin()[i] < min_corner_[i]) {
          quadrant[i] = LEFT;
          candidate_plane[i] = min_corner_[i];
          inside = false;
        }else if (ray.origin()[i] > max_corner_[i]) {
          quadrant[i] = RIGHT;
          candidate_plane[i] = max_corner_[i];
          inside = false;
        }else	{
          quadrant[i] = MIDDLE;
        }

      /* Ray origin inside bounding box */
      if(inside)	{
        return 0;
      }

      /* Calculate T distances to candidate planes */
      for (int i = 0; i < num_dim; i++)
        if (quadrant[i] != MIDDLE && ray.direction()[i] !=0.)
          max_T[i] = (candidate_plane[i]-ray.origin()[i]) / ray.direction()[i];
        else
          max_T[i] = -1.;

      /* Get largest of the max_T's for final choice of intersection */
      which_plane = 0;
      for (int i = 1; i < num_dim; i++)
        if (max_T[which_plane] < max_T[i])
          which_plane = i;

      /* Check final candidate actually inside box */
      if (max_T[which_plane] < 0.f) return dist;
      for (int i = 0; i < num_dim; i++)
        if (which_plane != i) {
          hit_point[i] = ray.origin()[i] + max_T[which_plane] *ray.direction()[i];
          if (hit_point[i] < min_corner_[i] || hit_point[i] > max_corner_[i])
            return dist;
        } else {
          hit_point[i] = candidate_plane[i];
        }

      dist = (hit_point - ray.origin()).norm();
      return (dist < SENSOR_LIMIT) ? dist : 0;
    }
  };

  Eigen::Vector3f center() {return center_;}
  Eigen::Vector3f dim() {return dim_;}
  Eigen::Vector3f min_corner() {return min_corner_;}
  Eigen::Vector3f max_corner() {return max_corner_;}

private:
  Eigen::Vector3f center_;
  Eigen::Vector3f dim_;
  Eigen::Vector3f min_corner_;
  Eigen::Vector3f max_corner_;
};

struct generate_depth_image {
public:
  generate_depth_image() {};
  generate_depth_image(float* depth_image, std::vector<obstacle*> obstacles)
      : depth_image_(depth_image), obstacles_(obstacles) {};

  void operator()(camera_parameter camera_parameter) {
    float focal_length_pix = camera_parameter.focalLengthPix();
    ray ray(camera_parameter);
    int image_width = camera_parameter.imageSize().x();
    int image_height = camera_parameter.imageSize().y();

    for (int u = 0; u < image_width; u++) {
      for (int v = 0; v < image_height; v++) {
        ray(u,v);
        float dist(SENSOR_LIMIT);
        for (std::vector<obstacle*>::iterator obstacle = obstacles_.begin(); obstacle != obstacles_.end(); ++obstacle) {
          float dist_tmp = (*obstacle)->intersect(ray);
          if (dist_tmp < dist)
            dist = dist_tmp;
        }

        float regularisation = std::sqrt(1 + se::math::sq(std::abs(u + 0.5 - image_width/2) / focal_length_pix)
                                         + se::math::sq(std::abs(v + 0.5 - image_height/2) / focal_length_pix));
        float depth = dist/regularisation;
        if(NOISE) {
          static std::mt19937 gen{1};
          std::normal_distribution<> d(0, 0.004*depth*depth);
          depth_image_[u + v*camera_parameter.imageSize().x()] = depth + d(gen);
        }
        else
          depth_image_[u + v*camera_parameter.imageSize().x()] = depth;
      }
    }
  }

private:
  float* depth_image_;
  std::vector<obstacle*> obstacles_;
};

class DenseAllocation : public ::testing::Test {
protected:
  virtual void SetUp() {
    size_ = 512;                             // 512 x 512 x 512 voxel^3
    voxel_dim_ = 0.005;                      // 5 mm/voxel
    doubling_factor_ = 2;                    // Travelled distance to node size ratio before doubling the allocation size
    max_allocation_size_ = 4*OctreeT::blockSide;
    dim_ = size_ * voxel_dim_;               // [m^3]
    oct_.init(size_, dim_);
    Eigen::Vector2i image_size(640, 480);    // width x height
    Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
    float FOV = 90;                          // [deg]
    camera_parameter_ = camera_parameter(FOV, image_size, camera_pose);
    band_ = 0.01;                            // [m]

    // Allocate memory for depth image
    depth_image_ =
        (float *) malloc(sizeof(float) * image_size.x() * image_size.y());
  }

  float* depth_image_;
  camera_parameter camera_parameter_;

  typedef se::Octree<MultiresOFusion> OctreeT;
  OctreeT oct_;
  int size_;
  float voxel_dim_;
  int doubling_factor_;
  int max_allocation_size_;
  float dim_;
  float band_;
  generate_depth_image generate_depth_image_;
  std::vector<se::key_t> allocation_list_;
  std::vector<se::key_t> free_space_list_;
  std::vector<se::key_t> parent_list_;
  std::vector<se::VoxelBlock<MultiresOFusion>*> active_list_;
};

TEST_F(DenseAllocation, DenseBFusionAllocationSphere) {
  std::vector<obstacle*> spheres;

  // Allocate single sphere in world frame
  spheres.push_back(new sphere_obstacle(voxel_dim_*Eigen::Vector3f(size_, size_*1/2, size_*1/2), 0.5f));
  generate_depth_image_ = generate_depth_image(depth_image_, spheres);

  Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
  // Camera orientation
  Eigen::Matrix3f Rbc;
  Rbc << 0, 0, 1, -1, 0, 0, 0, -1, 0;
  Eigen::Matrix3f Rwb = Eigen::Matrix3f::Identity();
  camera_pose.topLeftCorner<3,3>()  = Rwb*Rbc;
  // Camera position
  camera_pose.topRightCorner<3,1>() = Eigen::Vector3f(0, size_/2, size_/2)*voxel_dim_;

  camera_parameter_.setPose(camera_pose);
  generate_depth_image_(camera_parameter_);

  int num_vox_per_pix = size_;
  size_t total = num_vox_per_pix * camera_parameter_.imageSize().x() *
                 camera_parameter_.imageSize().y();
  allocation_list_.reserve(total);

  size_t allocated = buildOctantList(allocation_list_.data(), allocation_list_.capacity(), oct_,
                                     camera_pose, camera_parameter_.K(), depth_image_, camera_parameter_.imageSize(),
                                     voxel_dim_, band_, doubling_factor_, max_allocation_size_);
  oct_.allocate(allocation_list_.data(), allocated);

  std::stringstream f_ply;
  f_ply << "./out/dense-bfusion-allocation-sphere-unittest.ply";
  se::print_octree(f_ply.str().c_str(), oct_);

  for (std::vector<obstacle*>::iterator sphere = spheres.begin(); sphere != spheres.end(); ++sphere) {
    free(*sphere);
  }
  free(depth_image_);
};

TEST_F(DenseAllocation, DenseInFrustumBFusionAllocationSphere) {
  std::vector<obstacle*> spheres;

  // Allocate single sphere in world frame
  spheres.push_back(new sphere_obstacle(voxel_dim_*Eigen::Vector3f(size_, size_*1/2, size_*1/2), 0.5f));
  generate_depth_image_ = generate_depth_image(depth_image_, spheres);

  Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
  // Camera orientation
  Eigen::Matrix3f Rbc;
  Rbc << 0, 0, 1, -1, 0, 0, 0, -1, 0;
  Eigen::Matrix3f Rwb = Eigen::Matrix3f::Identity();
  camera_pose.topLeftCorner<3,3>()  = Rwb*Rbc;
  // Camera position
  camera_pose.topRightCorner<3,1>() = Eigen::Vector3f(0, size_/2, size_/2)*voxel_dim_;

  camera_parameter_.setPose(camera_pose);
  generate_depth_image_(camera_parameter_);

  int num_vox_per_pix = size_;
  size_t total = num_vox_per_pix * camera_parameter_.imageSize().x() *
                 camera_parameter_.imageSize().y();
  allocation_list_.reserve(total);
  free_space_list_.reserve(total);

  auto start = std::chrono::system_clock::now();

  size_t allocated;
  size_t free_space;

  buildDenseOctantList(allocation_list_.data(), free_space_list_.data(), allocated, free_space,
                       allocation_list_.capacity(), oct_, camera_pose, camera_parameter_.K(), depth_image_,
                       camera_parameter_.imageSize(), voxel_dim_, band_, doubling_factor_, max_allocation_size_);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

  oct_.allocate(allocation_list_.data(), allocated);
  oct_.allocate(free_space_list_.data(), free_space);

  std::stringstream f_ply;
  f_ply << "/home/nils/workspace_/projects/supereight/se_denseslam/test/out/dense-in-frustum-bfusion-allocation-sphere-unittest.ply";
  se::print_octree(f_ply.str().c_str(), oct_);

  for (std::vector<obstacle*>::iterator sphere = spheres.begin(); sphere != spheres.end(); ++sphere) {
    free(*sphere);
  }
  free(depth_image_);
};

TEST_F(DenseAllocation, FreeSpaceAllocation) {
  std::vector<obstacle*> spheres;

  // Allocate single sphere in world frame
  spheres.push_back(new sphere_obstacle(voxel_dim_*Eigen::Vector3f(size_, size_*1/2, size_*1/2), 0.5f));
  generate_depth_image_ = generate_depth_image(depth_image_, spheres);

  Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
  // Camera orientation
  Eigen::Matrix3f Rbc;
  Rbc << 0, 0, 1, -1, 0, 0, 0, -1, 0;
  Eigen::Matrix3f Rwb = Eigen::Matrix3f::Identity();
  camera_pose.topLeftCorner<3,3>()  = Rwb*Rbc;
  // Camera position
  camera_pose.topRightCorner<3,1>() = Eigen::Vector3f(0, size_/2, size_/2)*voxel_dim_;

  camera_parameter_.setPose(camera_pose);
  generate_depth_image_(camera_parameter_);

  int num_vox_per_pix = size_;
  size_t total = num_vox_per_pix * camera_parameter_.imageSize().x() *
                 camera_parameter_.imageSize().y();
  allocation_list_.reserve(total);
  free_space_list_.reserve(total);

  auto start = std::chrono::system_clock::now();

  size_t allocated;
  size_t free_space;

  buildDenseOctantList(allocation_list_.data(), free_space_list_.data(), allocated, free_space,
                       allocation_list_.capacity(), oct_, camera_pose, camera_parameter_.K(), depth_image_,
                       camera_parameter_.imageSize(), voxel_dim_, band_, doubling_factor_, max_allocation_size_);
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

//  int frame = 1;

//  oct_.update_free_space(free_space_list_.data(), free_space, frame);

  std::stringstream f_ply;
  f_ply << "/home/nils/workspace_/projects/supereight/se_denseslam/test/out/free-space-allocation-sphere-unittest.ply";
  se::print_octree(f_ply.str().c_str(), oct_);

  for (std::vector<obstacle*>::iterator sphere = spheres.begin(); sphere != spheres.end(); ++sphere) {
    free(*sphere);
  }
  free(depth_image_);
};

TEST_F(DenseAllocation, DenseBFusionAllocationViaParentSphere) {
  std::vector<obstacle*> spheres;

  // Allocate single sphere in world frame
  spheres.push_back(new sphere_obstacle(voxel_dim_*Eigen::Vector3f(size_, size_*1/2, size_*1/2), 0.5f));
  generate_depth_image_ = generate_depth_image(depth_image_, spheres);

  Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
  // Camera orientation
  Eigen::Matrix3f Rbc;
  Rbc << 0, 0, 1, -1, 0, 0, 0, -1, 0;
  Eigen::Matrix3f Rwb = Eigen::Matrix3f::Identity();
  camera_pose.topLeftCorner<3,3>()  = Rwb*Rbc;
  // Camera position
  camera_pose.topRightCorner<3,1>() = Eigen::Vector3f(0, size_/2, size_/2)*voxel_dim_;

  camera_parameter_.setPose(camera_pose);
  generate_depth_image_(camera_parameter_);

  int num_vox_per_pix = size_;
  size_t total = num_vox_per_pix * camera_parameter_.imageSize().x() *
                 camera_parameter_.imageSize().y();
  parent_list_.reserve(total);

  size_t allocated = buildParentOctantList(parent_list_.data(), parent_list_.capacity(), oct_,
                                           camera_pose, camera_parameter_.K(), depth_image_, camera_parameter_.imageSize(),
                                           voxel_dim_, band_, doubling_factor_, max_allocation_size_);
  oct_.allocateViaParent(parent_list_.data(), allocated);

  std::stringstream f_ply;
  f_ply << "./out/dense-bfusion-allocation-via-parent-sphere-unittest.ply";
  se::print_octree(f_ply.str().c_str(), oct_);

  for (std::vector<obstacle*>::iterator sphere = spheres.begin(); sphere != spheres.end(); ++sphere) {
    free(*sphere);
  }
  free(depth_image_);
};

TEST_F(DenseAllocation, DenseBFusionAllocationViaParentBox) {
  std::vector<obstacle*> boxes;

  // Allocate single sphere in world frame
  boxes.push_back(new box_obstacle(voxel_dim_*Eigen::Vector3f(size_, size_*1/2, size_*1/2), 0.5, 0.5, 0.5));
  generate_depth_image_ = generate_depth_image(depth_image_, boxes);

  Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
  // Camera orientation
  Eigen::Matrix3f Rbc;
  Rbc << 0, 0, 1, -1, 0, 0, 0, -1, 0;
  Eigen::Matrix3f Rwb = Eigen::Matrix3f::Identity();
  camera_pose.topLeftCorner<3,3>()  = Rwb*Rbc;
  // Camera position
  camera_pose.topRightCorner<3,1>() = Eigen::Vector3f(0, size_/2, size_/2)*voxel_dim_;

  camera_parameter_.setPose(camera_pose);
  generate_depth_image_(camera_parameter_);

  int num_vox_per_pix = size_;
  size_t total = num_vox_per_pix * camera_parameter_.imageSize().x() *
                 camera_parameter_.imageSize().y();
  parent_list_.reserve(total);

  size_t allocated = buildParentOctantList(parent_list_.data(), parent_list_.capacity(), oct_,
                                           camera_pose, camera_parameter_.K(), depth_image_, camera_parameter_.imageSize(),
                                           voxel_dim_, band_, doubling_factor_, max_allocation_size_);
  oct_.allocateViaParent(parent_list_.data(), allocated);

  std::stringstream f_ply;
  f_ply << "./out/dense-bfusion-allocation-via-parent-box-unittest.ply";
  se::print_octree(f_ply.str().c_str(), oct_);

  for (std::vector<obstacle*>::iterator box = boxes.begin(); box != boxes.end(); ++box) {
    free(*box);
  }
  free(depth_image_);
};