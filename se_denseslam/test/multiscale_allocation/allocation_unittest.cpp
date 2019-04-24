#include <se/octant_ops.hpp>
#include <se/octree.hpp>
#include <se/algorithms/balancing.hpp>
#include <se/functors/axis_aligned_functor.hpp>
#include <se/algorithms/filter.hpp>
#include <se/volume_traits.hpp>
#include <se/io/vtk-io.h>
#include <se/io/ply_io.hpp>
#include <sophus/se3.hpp>
#include <se/utils/math_utils.h>
#include "se/node.hpp"
#include "se/functors/data_handler.hpp"
#include <random>
#include <functional>
#include <gtest/gtest.h>
#include <vector>
#include <stdio.h>

// Truncation distance and maximum weight
#define MAX_DIST 2.f
#define MAX_WEIGHT 5
#define MU 0.1f

// Fusion level,
// 0-3 are the according levels in the voxel_block
// 4 is multilevel fusion
#define SCALE 4

#define ALLOCATION_FACTOR 4

// Returned distance when ray doesn't intersect sphere
#define SENSOR_LIMIT 7

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
  float focal_length_pix() {return focal_length_pix_;}
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
    float dist(SENSOR_LIMIT);
    Eigen::Vector3f oc = ray.origin() - center_;
    float a = ray.direction().dot(ray.direction());
    float b = 2.0 * oc.dot(ray.direction());
    float c = oc.dot(oc) - radius_*radius_;
    float discriminant = b*b - 4*a*c;
    if (discriminant >= 0) {
      float dist_tmp = (-b - sqrt(discriminant))/(2.0*a);
      if (dist_tmp < dist)
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
    float dist(SENSOR_LIMIT);
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
      return dist;
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
    float focal_length_pix = camera_parameter.focal_length_pix();
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

inline float compute_scale(const Eigen::Vector3f& vox, 
                    const Eigen::Vector3f& twc,
                    const float scaled_pix,
                    const float voxelsize) {
  const float dist = (voxelsize * vox - twc).norm();
  const float pix_size = dist * scaled_pix;
  int scale = std::min(std::max(0, int(log2(pix_size/voxelsize + 0.5))),
                       3);
  return scale;
}

template <typename T>
void propagate_down(se::VoxelBlock<T>* block, const int scale) {
  const Eigen::Vector3i base = block->coordinates();
  const int side = se::VoxelBlock<T>::side;
  for(int curr_scale = scale; curr_scale > 0; --curr_scale) {
    const int stride = 1 << curr_scale;
    for (int z = 0; z < side; z += stride)
      for (int y = 0; y < side; y += stride)
        for (int x = 0; x < side; x += stride) {
          const Eigen::Vector3i parent = base + Eigen::Vector3i(x, y, z);
          auto data = block->data(parent, curr_scale);

          float virt_sample = data.delta*float(data.y - 1) + data.x;
          typedef voxel_traits<T> traits_type;
          typedef typename traits_type::value_type value_type;
          value_type curr_list[8];
          Eigen::Vector3i vox_list[8];
          float delta_sum(0);

          int idx = 0;
          for (int k = 0; k < stride; k += stride/2)
            for (int j = 0; j < stride; j += stride/2)
              for (int i = 0; i < stride; i += stride/2) {
                vox_list[idx] = parent + Eigen::Vector3i(i, j, k);
                auto curr = block->data(vox_list[idx], curr_scale -1);
                // Calculate non normalized child delta
                curr.delta = virt_sample - curr.x;
                delta_sum += curr.delta;
                curr_list[idx] = curr;
                ++idx;
              }

          for (int i = 0; i < 8; i++) {
            // Update delta_x
            if (delta_sum != 0)
              curr_list[i].delta = data.delta*curr_list[i].delta/delta_sum*8;

            // Update x
            curr_list[i].x = curr_list[i].y == 0 ? data.x :
                curr_list[i].x += curr_list[i].delta;

            // Update weight (with 0 <= y <= MAX_WEIGHT)
            curr_list[i].y = curr_list[i].y == 0 ? data.y :
                             std::min(curr_list[i].y + data.delta_y, MAX_WEIGHT);
            curr_list[i].delta_y =data.delta_y;

            block->data(vox_list[i], curr_scale - 1, curr_list[i]);
          }

          data.delta = 0;
          data.delta_y = 0;
          block->data(parent, curr_scale, data);
        }
  }
}

template <typename T>
void propagate_up(se::VoxelBlock<T>* block, const int scale) {
  const Eigen::Vector3i base = block->coordinates();
  const int side = se::VoxelBlock<T>::side;
  for(int curr_scale = scale; curr_scale < se::math::log2_const(side); ++curr_scale) {
    const int stride = 1 << (curr_scale + 1);
    for (int z = 0; z < side; z += stride)
      for (int y = 0; y < side; y += stride)
        for (int x = 0; x < side; x += stride) {
          const Eigen::Vector3i curr = base + Eigen::Vector3i(x, y, z);

          float mean = 0;
          int num_samples = 0;
          float weight = 0;
          for (int k = 0; k < stride; k += stride/2)
            for (int j = 0; j < stride; j += stride/2)
              for (int i = 0; i < stride; i += stride/2) {
                auto tmp = block->data(curr + Eigen::Vector3i(i, j, k), curr_scale);
                if (tmp.y != 0) {
                  mean += tmp.x;
                  weight += tmp.y;
                  num_samples++;
                }
              }

          auto data = block->data(curr, curr_scale + 1);

          if (num_samples != 0) {
            // Update SDF value to mean of its children
            mean /= num_samples;
            data.x = mean;

            // Update weight (round up if > 0.5, round down otherwise)
            weight /= num_samples;
            data.y = ceil(weight);
          } else {
            data = voxel_traits<MultiresSDF>::initValue();
          }

          data.delta = 0;
          data.delta_y = 0;
          block->data(curr, curr_scale + 1, data);
        }
  }
}

template <typename T>
void foreach(float voxelsize, std::vector<se::VoxelBlock<T>*> active_list,
             camera_parameter camera_parameter, float* depth_image) {
  const int n = active_list.size();
  for(int i = 0; i < n; ++i) {
    se::VoxelBlock<T>* block = active_list[i];
    const Eigen::Vector3i base = block->coordinates();
    const int side = se::VoxelBlock<T>::side;

    const Eigen::Matrix4f Tcw = (camera_parameter.Twc()).inverse();
    const Eigen::Matrix3f Rcw = Tcw.topLeftCorner<3,3>();
    const Eigen::Vector3f tcw = Tcw.topRightCorner<3,1>();
    const Eigen::Matrix4f K = camera_parameter.K();
    const Eigen::Vector2i image_size = camera_parameter.imageSize();
    const float scaled_pix = (camera_parameter.K().inverse() * (Eigen::Vector3f(1, 0 ,1) - Eigen::Vector3f(0, 0, 1)).homogeneous()).x();

    // Calculate the maximum uncertainty possible
    int scale = compute_scale((base + Eigen::Vector3i::Constant(side/2)).cast<float>(),
                               tcw, scaled_pix, voxelsize);
    if (SCALE != 4)
      scale = SCALE;
    float stride = std::max(int(pow(2,scale)),1);
    for(float z = stride/2; z < side; z += stride) {
      for (float y = stride/2; y < side; y += stride) {
        for (float x = stride/2; x < side; x += stride) {
          const Eigen::Vector3f node_w = base.cast<float>() + Eigen::Vector3f(x, y, z);
          const Eigen::Vector3f node_c = Rcw * (voxelsize * node_w)+ tcw;
          auto data = block->data(node_w.cast<int>(), scale);
          if (node_c.z() < 0.0001f)
            continue;
          const Eigen::Vector3f pixel_homo = K.topLeftCorner<3, 3>() * node_c;
          const float inverse_depth = 1.f / pixel_homo.z();
          const Eigen::Vector2f pixel = Eigen::Vector2f(
              pixel_homo.x() * inverse_depth,
              pixel_homo.y() * inverse_depth);
          if (pixel(0) < 0.5f || pixel(0) > image_size.x() - 1.5f ||
              pixel(1) < 0.5f || pixel(1) > image_size.y() - 1.5f)
            continue;

          float depth = depth_image[int(pixel.x()) + image_size.x()*int(pixel.y())];
          const float diff = (depth - node_c.z()) * std::sqrt( 1 + se::math::sq(node_c.x() / node_c.z()) + se::math::sq(node_c.y() / node_c.z()));
          if (diff > -MU) {
            const float sample = fminf(MAX_DIST, diff);

            // Make sure that the max weight isn't greater than MAX_WEIGHT (i.e. y + 1)
            data.y = std::min(data.y, MAX_WEIGHT - 1);

            // Update SDF value
            data.delta = (sample - data.x)/(data.y + 1);
            data.x = (data.x * data.y + sample)/(data.y + 1);

            // Update weight
            data.delta_y++;
            data.y = data.y + 1;

            block->data(node_w.cast<int>(), scale, data);
          }
        }
      }
    }

    propagate_down(block, scale);
    propagate_up(block, 0);
  }
}

template <typename FieldType,
    template <typename> class OctreeT,
    typename HashType>
size_t buildOctantList(HashType*              allocation_list,
                       size_t                 reserved,
                       OctreeT<FieldType>&    oct,
                       camera_parameter       camera_parameter,
                       const float*           depth_image,
                       const float            voxel_size,
                       const float            band) {
  // Initalize map parameter
  const int size = oct.size();
  const int max_level = log2(size);
  const int leaves_level = max_level - se::math::log2_const(OctreeT<FieldType>::blockSide);

  // Inverse voxel size
  const float inv_voxel_size = 1.f/voxel_size;

  // Compute ray computation for given camera pose
  ray ray(camera_parameter);

  int image_width = camera_parameter.imageSize().x();
  int image_height = camera_parameter.imageSize().y();
  unsigned int voxelCount = 0;

  Eigen::Vector3f allocation_end = camera_parameter.twc();

  for (int u = 0; u < image_width; u++) {
    for (int v = 0; v < image_height; v++) {
      // Access depth value; Continue if not valid (i.e. 0)
      const float depth = depth_image[u + v * image_width];
      if (depth == 0)
        continue;

      // Compute direction and vortex in world frame for given pixel (u,v)
      ray(u, v, depth);
      Eigen::Vector3f vertex_w = ray.vertex_w();

      // Define direction as normalized path from surface to camera origin (i.e. opposite direction to ray)
      Eigen::Vector3f direction = -ray.direction();

      // Range := Distance along the ray from surface to camera centre [m]
      float range = ray.range();

      // Distance := Distance along the ray from band/2 behind the surface to camera centre [m]
      float distance = inv_voxel_size*(range + band/2);

      // Starting point for allocation
      Eigen::Vector3f allocation_origin = vertex_w - band/2*direction;

      // Initialization
      int side = se::VoxelBlock<FieldType>::side;
      int allocation_size = 2*side;

      // Allocate free_space at higher resolution
      if (0.999*SENSOR_LIMIT < depth) {
        int free_space_factor = 4;
        allocation_size = free_space_factor*allocation_size;
      }

      Eigen::Vector3f curr_pos = allocation_origin;
      Eigen::Vector3f curr_pos_scaled = inv_voxel_size*curr_pos;
      Eigen::Vector3i curr_node = allocation_size*(((curr_pos_scaled).array().floor())/allocation_size).cast<int>();
      Eigen::Vector3f frac = (curr_pos_scaled - curr_node.cast<float>())/allocation_size;
      int allocation_level = max_level - log2(allocation_size);

      //Current state of T
      Eigen::Vector3f T_max;

      // Increment/Decrement of voxel value along the ray (-1 or +1)
      Eigen::Vector3i step_base;
      Eigen::Vector3i step;

      // Time need to pass one voxel in x, y and z direction [voxel]
      Eigen::Vector3f delta_T = allocation_size/direction.array().abs();

      // Initalize T
      if(direction.x() < 0) {
        step_base.x()  = -1;
        T_max.x() = frac.x() * delta_T.x();
      }
      else {
        step_base.x() = 1;
        T_max.x() = (1 - frac.x()) * delta_T.x();
      }
      if(direction.y() < 0) {
        step_base.y()  = -1;
        T_max.y() = frac.y() * delta_T.y();
      }
      else {
        step_base.y() = 1;
        T_max.y() = (1 - frac.y()) * delta_T.y();
      }
      if(direction.z() < 0) {
        step_base.z()  = -1;
        T_max.z() = frac.z() * delta_T.z();
      }
      else {
        step_base.z() = 1;
        T_max.z() = (1 - frac.z()) * delta_T.z();
      }

      step = allocation_size*step_base;

      float travelled = 0;

      do {
        if ((curr_node.x() < size)
            && (curr_node.y() < size)
            && (curr_node.z() < size)
            && (curr_node.x() >= 0)
            && (curr_node.y() >= 0)
            && (curr_node.z() >= 0)) {
          auto node_ptr = oct.fetch_octant(curr_node.x(), curr_node.y(), curr_node.z(),
                                           allocation_level);
          if (!node_ptr) {
            HashType key = oct.hash(curr_node.x(), curr_node.y(), curr_node.z(),
                                    std::min(allocation_level, leaves_level));
            unsigned int idx = voxelCount++;
            if(idx < reserved) {
              allocation_list[idx] = key;
            }
          } else if (allocation_level >= leaves_level) {
            static_cast<se::VoxelBlock<FieldType>*>(node_ptr)->active(true);
          }
        }

        // Update allocation
        // Double allocation size every time the allocation distance from the surface is bigger than ALLOCATION_FACTOR * allocation_size
        if ((travelled - inv_voxel_size*band/2) > ALLOCATION_FACTOR*allocation_size) {
          allocation_size = 2*allocation_size;

          // Update current position along the ray where
          // allocation_origin [m]
          // travelled*direction [voxel]
          curr_pos_scaled = inv_voxel_size*allocation_origin + travelled*direction;

          // Update curr_node to match the node size
          curr_node = allocation_size*(((curr_node).array().floor())/allocation_size);

          // Compute fractions of current position in new node volume
          frac = (curr_pos_scaled - curr_node.cast<float>())/allocation_size;

          // Reduce allocation level to parent level
          allocation_level -= 1;
          delta_T = allocation_size/direction.array().abs();
          step = allocation_size*step_base;

          if(direction.x() < 0) {
            T_max.x() = travelled + frac.x() * delta_T.x();
          }
          else {
            T_max.x() = travelled + (1 - frac.x()) * delta_T.x();
          }
          if(direction.y() < 0) {
            T_max.y() = travelled + frac.y() * delta_T.y();
          }
          else {
            T_max.y() = travelled + (1 - frac.y()) * delta_T.y();
          }
          if(direction.z() < 0) {
            T_max.z() = travelled + frac.z() * delta_T.z();
          }
          else {
            T_max.z() = travelled + (1 - frac.z()) * delta_T.z();
          }
        }

        if (T_max.x() < T_max.y()) {
          if (T_max.x() < T_max.z()) {
            travelled = T_max.x();
            curr_node.x() += step.x();
            T_max.x() += delta_T.x();
          } else {
            travelled = T_max.z();
            curr_node.z() += step.z();
            T_max.z() += delta_T.z();
          }
        } else {
          if (T_max.y() < T_max.z()) {
            travelled = T_max.y();
            curr_node.y() += step.y();
            T_max.y() += delta_T.y();
          } else {
            travelled = T_max.z();
            curr_node.z() += step.z();
            T_max.z() += delta_T.z();
          }
        }
      } while (0 < (distance - travelled));
    }
  }
  return (size_t) voxelCount >= reserved ? reserved : (size_t) voxelCount;
}

template <typename T>
std::vector<se::VoxelBlock<MultiresSDF>*> buildActiveList(se::Octree<T>& map, camera_parameter camera_parameter, float voxel_size) {
  const se::MemoryPool<se::VoxelBlock<MultiresSDF> >& block_array =
      map.getBlockBuffer();
  for(unsigned int i = 0; i < block_array.size(); ++i) {
    block_array[i]->active(false);
  }

  const Eigen::Matrix4f K = camera_parameter.K();
  const Eigen::Matrix4f Twc = camera_parameter.Twc();
  const Eigen::Matrix4f Tcw = (camera_parameter.Twc()).inverse();
  std::vector<se::VoxelBlock<MultiresSDF>*> active_list;
  auto in_frustum_predicate =
      std::bind(se::algorithms::in_frustum<se::VoxelBlock<MultiresSDF>>, std::placeholders::_1,
                voxel_size, K*Tcw, camera_parameter.imageSize());
  se::algorithms::filter(active_list, block_array, in_frustum_predicate);
  return active_list;
}

class MultiscaleAllocation : public ::testing::Test {
protected:
  virtual void SetUp() {
    size_ = 512;                              // 512 x 512 x 512 voxel^3
    voxel_size_ = 0.005;                      // 5 mm/voxel
    dim_ = size_ * voxel_size_;         // [m^3]
    oct_.init(size_, dim_);
    Eigen::Vector2i image_size(640, 480);    // width x height
    Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
    float FOV = 90; // [deg]
    camera_parameter_ = camera_parameter(90, image_size, camera_pose);
    band_ = 0.01;

    // Generate depth image
    depth_image_ =
        (float *) malloc(sizeof(float) * image_size.x() * image_size.y());
  }

  float* depth_image_;
  camera_parameter camera_parameter_;

  typedef se::Octree<MultiresSDF> OctreeT;
  OctreeT oct_;
  int size_;
  float voxel_size_;
  float dim_;
  float band_;
  generate_depth_image generate_depth_image_;
  std::vector<se::key_t> allocation_list_;
  std::vector<se::VoxelBlock<MultiresSDF>*> active_list_;
};

TEST_F(MultiscaleAllocation, BoxTranslation) {
  std::vector<obstacle*> boxes;

  // Allocate boxes in world frame
  boxes.push_back(new box_obstacle(voxel_size_*Eigen::Vector3f(size_*1/2, size_*1/4, size_/2), voxel_size_*Eigen::Vector3f(size_*1/4, size_*1/4, size_/4)));
  boxes.push_back(new box_obstacle(voxel_size_*Eigen::Vector3f(size_*1/2, size_*3/4, size_/2), voxel_size_*Eigen::Vector3f(size_*1/4, size_*1/4, size_/4)));
  generate_depth_image_ = generate_depth_image(depth_image_, boxes);

  int frames = FRAMES;
  for (int frame = 0; frame <= frames; frame++) {
    std::cout << "Processing frame " << frame << "/" << frames << "." << std::endl;
    Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f Rbc;
    Rbc << 0, 0, 1, -1, 0, 0, 0, -1, 0;

    Eigen::Matrix3f Rwb = Eigen::Matrix3f::Identity();

    camera_pose.topLeftCorner<3,3>()  = Rwb*Rbc;

    camera_pose.topRightCorner<3,1>() = (Rwb*Eigen::Vector3f(-(size_/2 + frame*size_/8), 0, size_/2) + Eigen::Vector3f(size_/2, size_/2, 0))*voxel_size_;

    camera_parameter_.setPose(camera_pose);
    generate_depth_image_(camera_parameter_);

    int num_vox_per_pix = float(size_);
    size_t total = num_vox_per_pix * camera_parameter_.imageSize().x() *
        camera_parameter_.imageSize().y();
    allocation_list_.reserve(total);

    size_t allocated = buildOctantList(allocation_list_.data(), allocation_list_.capacity(), oct_, camera_parameter_, depth_image_, voxel_size_, band_);
    oct_.allocate_multiscale(allocation_list_.data(), allocated);
    active_list_ = buildActiveList(oct_, camera_parameter_, voxel_size_);
    foreach(voxel_size_, active_list_, camera_parameter_, depth_image_);
    std::stringstream f_vtk;

    f_vtk << "/home/nils/workspace_ptp/catkin_ws/src/probabilistic_trajectory_planning_ros/ext/probabilistic_trajectory_planning/src/ext/supereight/se_denseslam/test/out/box_allocation-unittest-" + std::to_string(frame) +".vtk";

    save3DSlice(oct_,
                Eigen::Vector3i(0, 0, oct_.size()/2),
                Eigen::Vector3i(oct_.size(), oct_.size(), oct_.size()/2 + 1),
                [](const auto& val) { return val.x; }, f_vtk.str().c_str());

    std::stringstream f_ply;
    f_ply << "/home/nils/workspace_ptp/catkin_ws/src/probabilistic_trajectory_planning_ros/ext/probabilistic_trajectory_planning/src/ext/supereight/se_denseslam/test/out/box_allocation-unittest-" + std::to_string(frame) + ".ply";
    se::print_octree(f_ply.str().c_str(), oct_);
  }


  for (std::vector<obstacle*>::iterator box = boxes.begin(); box != boxes.end(); ++box) {
    free(*box);
  }
  free(depth_image_);
}

TEST_F(MultiscaleAllocation, SphereTranslation) {
  std::vector<obstacle*> spheres;

  // Allocate spheres in world frame
  sphere_obstacle* sphere_close = new sphere_obstacle(voxel_size_*Eigen::Vector3f(size_*1/8, size_*2/3, size_/2), 0.3f);
  sphere_obstacle* sphere_far   = new sphere_obstacle(voxel_size_*Eigen::Vector3f(size_*7/8, size_*1/3, size_/2), 0.3f);
  spheres.push_back(sphere_close);
  spheres.push_back(sphere_far);
  generate_depth_image_ = generate_depth_image(depth_image_, spheres);

  int frames = FRAMES;
  for (int frame = 0; frame <= frames; frame++) {
    std::cout << "Processing frame " << frame << "/" << frames << "." << std::endl;
    Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f Rbc;
    Rbc << 0, 0, 1, -1, 0, 0, 0, -1, 0;

    Eigen::Matrix3f Rwb = Eigen::Matrix3f::Identity();

    camera_pose.topLeftCorner<3,3>()  = Rwb*Rbc;

    camera_pose.topRightCorner<3,1>() = (Rwb*Eigen::Vector3f(-(size_/2 + frame*size_/8), 0, size_/2) + Eigen::Vector3f(size_/2, size_/2, 0))*voxel_size_;

    camera_parameter_.setPose(camera_pose);
    generate_depth_image_(camera_parameter_);

    int num_vox_per_pix = float(size_);
    size_t total = num_vox_per_pix * camera_parameter_.imageSize().x() *
        camera_parameter_.imageSize().y();
    allocation_list_.reserve(1.5*total);

    size_t allocated = buildOctantList(allocation_list_.data(), allocation_list_.capacity(), oct_, camera_parameter_, depth_image_, voxel_size_, band_);
    oct_.allocate_multiscale(allocation_list_.data(), allocated);
    active_list_ = buildActiveList(oct_, camera_parameter_, voxel_size_);
    foreach(voxel_size_, active_list_, camera_parameter_, depth_image_);
    std::stringstream f_vtk;

    f_vtk << "/home/nils/workspace_ptp/catkin_ws/src/probabilistic_trajectory_planning_ros/ext/probabilistic_trajectory_planning/src/ext/supereight/se_denseslam/test/out/sphere_allocation-unittest-" + std::to_string(frame) +".vtk";

    save3DSlice(oct_,
                Eigen::Vector3i(0, 0, oct_.size()/2),
                Eigen::Vector3i(oct_.size(), oct_.size(), oct_.size()/2 + 1),
                [](const auto& val) { return val.x; }, f_vtk.str().c_str());

    std::stringstream f_ply;
    f_ply << "/home/nils/workspace_ptp/catkin_ws/src/probabilistic_trajectory_planning_ros/ext/probabilistic_trajectory_planning/src/ext/supereight/se_denseslam/test/out/sphere_allocation-unittest-" + std::to_string(frame) + ".ply";
    se::print_octree(f_ply.str().c_str(), oct_);
  }

  for (std::vector<obstacle*>::iterator sphere = spheres.begin(); sphere != spheres.end(); ++sphere) {
    free(*sphere);
  }
  free(depth_image_);
}

TEST_F(MultiscaleAllocation, AllocateLevelMultiscale) {
  allocation_list_.reserve(1);
  const int max_level = log2(size_);
  const int leaves_level = max_level - se::math::log2_const(se::VoxelBlock<MultiresSDF>::side);
  se::key_t key = oct_.hash(511, 511, 511, leaves_level);
  allocation_list_[0] = key;
  oct_.allocate_multiscale(allocation_list_.data(), 1);

  se::print_octree("/home/nils/workspace_ptp/catkin_ws/src/probabilistic_trajectory_planning_ros/ext/probabilistic_trajectory_planning/src/ext/supereight/se_denseslam/test/out/allocate_level_multiscale-unittest.ply", oct_);

}