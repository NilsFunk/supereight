/*
 *
 * Copyright 2016 Emanuele Vespa, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * */
#ifndef BFUSION_ALLOC_H
#define BFUSION_ALLOC_H
#include <se/utils/math_utils.h>

/* Compute step size based on distance travelled along the ray */
static inline float compute_stepsize(const float dist_travelled,
                                     const float hf_band,
                                     const float voxelSize) {
  float new_step;
  float half = hf_band * 0.5f;
  if (dist_travelled < hf_band) {
    new_step = voxelSize;
  } else if (dist_travelled < hf_band + half) {
    new_step = 10.f * voxelSize;
  } else {
    new_step = 30.f * voxelSize;
  }
  return new_step;
}

/* Compute octree level given a step size */
static inline int step_to_depth(const float step,
                                const int max_depth,
                                const float voxelsize) {
  return static_cast<int>(floorf(std::log2f(voxelsize/step)) + max_depth);
}

template <typename FieldType,
          template <typename> class OctreeT,
          typename HashType,
          typename StepF, typename DepthF>
size_t buildOctantList(HashType*              allocationList,
                       size_t                 reserved,
                       OctreeT<FieldType>&    map_index,
                       const Eigen::Matrix4f& pose,
                       const Eigen::Matrix4f& K,
                       const float*           depthmap,
                       const Eigen::Vector2i& imageSize,
                       const float            voxelSize,
                       StepF                  compute_stepsize,
                       DepthF                 step_to_depth,
                       const float            band) {

  const float inverseVoxelSize = 1.f/voxelSize;
  Eigen::Matrix4f invK = K.inverse();
  const Eigen::Matrix4f kPose = pose * invK;
  const int size = map_index.size();
  const int max_depth = log2(size);
  const int leaves_depth = max_depth - se::math::log2_const(OctreeT<FieldType>::blockSide);

#ifdef _OPENMP
  std::atomic<unsigned int> voxelCount;
  std::atomic<unsigned int> leavesCount;
#else
  unsigned int voxelCount;
#endif

  const Eigen::Vector3f camera = pose.topRightCorner<3, 1>();
  voxelCount = 0;
#pragma omp parallel for
  for (int y = 0; y < imageSize.y(); ++y) {
    for (int x = 0; x < imageSize.x(); ++x) {
      if(depthmap[x + y*imageSize.x()] == 0)
        continue;
      int tree_depth = max_depth;
      float stepsize = voxelSize;
      const float depth = depthmap[x + y*imageSize.x()];
      Eigen::Vector3f worldVertex = (kPose * Eigen::Vector3f((x + 0.5f) * depth,
            (y + 0.5f) * depth, depth).homogeneous()).head<3>();

      Eigen::Vector3f direction = (camera - worldVertex).normalized();
      const Eigen::Vector3f origin = worldVertex - (band * 0.5f) * direction;
      const float dist = (camera - origin).norm();
      Eigen::Vector3f step = direction*stepsize;

      Eigen::Vector3f voxelPos = origin;
      float travelled = 0.f;
      for (; travelled < dist; travelled += stepsize) {

        Eigen::Vector3f voxelScaled = (voxelPos * inverseVoxelSize).array().floor();
        if ((voxelScaled.x() < size)
            && (voxelScaled.y() < size)
            && (voxelScaled.z() < size)
            && (voxelScaled.x() >= 0)
            && (voxelScaled.y() >= 0)
            && (voxelScaled.z() >= 0)) {
          const Eigen::Vector3i voxel = voxelScaled.cast<int>();
          auto node_ptr = map_index.fetch_octant(voxel.x(), voxel.y(), voxel.z(),
              tree_depth);
          if (!node_ptr) {
            HashType k = map_index.hash(voxel.x(), voxel.y(), voxel.z(),
                std::min(tree_depth, leaves_depth));
            unsigned int idx = voxelCount++;
            if(idx < reserved) {
              allocationList[idx] = k;
            }
          } else if (tree_depth >= leaves_depth) {
            static_cast<se::VoxelBlock<FieldType>*>(node_ptr)->active(true);
          }
        }
        stepsize = compute_stepsize(travelled, band, voxelSize);
        tree_depth = step_to_depth(stepsize, max_depth, voxelSize);

        step = direction*stepsize;
        voxelPos +=step;
      }
    }
  }
  return (size_t) voxelCount >= reserved ? reserved : (size_t) voxelCount;
}

template <typename FieldType,
    template <typename> class OctreeT,
    typename HashType>
size_t buildOctantList(HashType*               allocation_list,
                       size_t                  reserved_keys,
                       OctreeT<FieldType>&     oct,
                       const Eigen::Matrix4f&  camera_pose,
                       const Eigen::Matrix4f&  K,
                       const float*            depthmap,
                       const Eigen::Vector2i&  image_size,
                       const float             voxel_dim,
                       const float             band,
                       const int               doubling_ratio,
                       int                     min_allocation_size) {
  // Create inverse voxel dimension, camera matrix and projection matrix
  const float inv_voxel_dim = 1.f/voxel_dim; // inv_voxel_dim := [m] to [voxel]; voxel_dim := [voxel] to [m]
  Eigen::Matrix4f inv_K = K.inverse();
  const Eigen::Matrix4f inv_P = camera_pose * inv_K;

  // Read map parameter
  const int   size = oct.size();
  const int   max_level = log2(size);
  const int   leaves_level = max_level - se::math::log2_const(OctreeT<FieldType>::blockSide);
  const int   side = se::VoxelBlock<FieldType>::side;
  const int   init_allocation_size = side;
  min_allocation_size = (min_allocation_size > init_allocation_size) ? min_allocation_size : init_allocation_size;

#ifdef _OPENMP
  std::atomic<unsigned int> voxel_count;
#else
  unsigned int voxel_count;
#endif
  // Camera position [m] in world frame
  const Eigen::Vector3f camera_position = camera_pose.topRightCorner<3, 1>();
  voxel_count = 0;
#pragma omp parallel for
  for (int y = 0; y < image_size.y(); ++y) {
    for (int x = 0; x < image_size.x(); ++x) {
      if(depthmap[x + y*image_size.x()] == 0)
        continue;
      const float depth = depthmap[x + y*image_size.x()];
      Eigen::Vector3f world_vertex = (inv_P * Eigen::Vector3f((x + 0.5f) * depth,
                                                              (y + 0.5f) * depth,
                                                              depth).homogeneous()).head<3>(); //  [m] in world frame

      // Vertex to camera direction in [-] (no unit) in world frame
      Eigen::Vector3f direction = (camera_position - world_vertex).normalized();

      // Position behind the surface in [m] in world frame
      const Eigen::Vector3f allocation_origin = world_vertex - (band * 0.5f) * direction;

      // Voxel/node traversal origin to camera distance in [voxel]
      const float distance = inv_voxel_dim*(camera_position - allocation_origin).norm();

      // Initialise side length in [voxel] of allocated node
      int allocation_size = init_allocation_size;
      int allocation_level = max_level - log2(allocation_size);

      Eigen::Vector3f curr_pos_m = allocation_origin;
      Eigen::Vector3f curr_pos_v = inv_voxel_dim*curr_pos_m;
      Eigen::Vector3i curr_node = allocation_size*(((curr_pos_v).array().floor())/allocation_size).cast<int>();
      // Fraction of the current position in [voxel] in the current node along the x-, y- and z-axis
      Eigen::Vector3f frac = (curr_pos_v - curr_node.cast<float>())/allocation_size;

      //Current state of T in [voxel]
      Eigen::Vector3f T_max;
      // Increment/Decrement of voxel value along the ray (-1 or +1)
      Eigen::Vector3i step_base;
      // Scaled step_base in [voxel]. Scaling factor will be the current allocation_size
      Eigen::Vector3i step;
      // Travelled distance needed in [voxel] to pass a voxel in x, y and z direction
      Eigen::Vector3f delta_T = allocation_size/direction.array().abs(); // [voxel]/[-]

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

      // Distance travelled in [voxel]
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
            unsigned const idx = voxel_count++;
            if(voxel_count <= reserved_keys) {
              allocation_list[idx] = key;
            }
          } else if (allocation_level >= leaves_level) {
            static_cast<se::VoxelBlock<FieldType>*>(node_ptr)->active(true);
          }
        }

        // Update allocation variables
        // Double allocation size every time the allocation distance from the surface is bigger than doubling_ratio * allocation_size
        if ((travelled - inv_voxel_dim*band/2) > doubling_ratio*allocation_size &&
            (travelled - inv_voxel_dim*band)   > 0 &&
            allocation_size < min_allocation_size) {
          allocation_size = 2*allocation_size;

          // Update current position along the ray where
          // allocation_origin in [m] and travelled*direction in [voxel]
          curr_pos_v = inv_voxel_dim*allocation_origin + travelled*direction;

          // Re-initialize the curr_node to match the allocation size
          curr_node = allocation_size*(((curr_node).array().floor())/allocation_size);

          // Compute fraction of the current position in [voxel] in the updated current node along the x-, y- and z-axis
          frac = (curr_pos_v - curr_node.cast<float>())/allocation_size;

          // Reduce allocation level to coarser level by 1
          allocation_level -= 1;

          // Re-initalize delta_T, T_max and step size according to new allocation_size
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

        // Traverse to closest face crossing of the voxel block/node (i.e. find minimum T_max)
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
  return (size_t) voxel_count >= reserved_keys ? reserved_keys : (size_t) voxel_count;
}

template <typename FieldType,
    template <typename> class OctreeT,
    typename HashType>
size_t buildParentOctantList(HashType*               parent_list,
                             size_t                  reserved_keys,
                             OctreeT<FieldType>&     oct,
                             const Eigen::Matrix4f&  camera_pose,
                             const Eigen::Matrix4f&  K,
                             const float*            depthmap,
                             const Eigen::Vector2i&  image_size,
                             const float             voxel_dim,
                             const float             band,
                             const int               doubling_ratio,
                             const int               min_allocation_size) {
  // Create inverse voxel dimension, camera matrix and projection matrix
  const float inv_voxel_dim = 1.f / voxel_dim; // inv_voxel_dim := [m] to [voxel]; voxel_dim := [voxel] to [m]
  Eigen::Matrix4f inv_K = K.inverse();
  const Eigen::Matrix4f inv_P = camera_pose * inv_K;

  // Read map parameter
  const int size = oct.size();
  const int max_level = log2(size);
  const int leaves_level = max_level - se::math::log2_const(OctreeT<FieldType>::blockSide);
  const int side = se::VoxelBlock<FieldType>::side;
  const int init_allocation_size = side;
  const int init_parent_size = 2 * init_allocation_size;
  const int min_parent_size = (min_allocation_size > init_allocation_size) ? 2 * min_allocation_size : init_parent_size;

#ifdef _OPENMP
  std::atomic<unsigned int> parent_count;
#else
  unsigned int parent_count;
#endif
  // Camera position [m] in world frame
  const Eigen::Vector3f camera_position = camera_pose.topRightCorner<3, 1>();
  parent_count = 0;
#pragma omp parallel for
  for (int y = 0; y < image_size.y(); ++y) {
    for (int x = 0; x < image_size.x(); ++x) {
      if (depthmap[x + y * image_size.x()] == 0)
        continue;
      const float depth = depthmap[x + y * image_size.x()];
      Eigen::Vector3f world_vertex = (inv_P * Eigen::Vector3f((x + 0.5f) * depth,
                                                              (y + 0.5f) * depth,
                                                              depth).homogeneous()).head<3>(); //  [m] in world frame

      // Vertex to camera direction in [-] (no unit) in world frame
      Eigen::Vector3f direction = (camera_position - world_vertex).normalized();

      // Position behind the surface in [m] in world frame
      const Eigen::Vector3f allocation_origin = world_vertex - (band * 0.5f) * direction;

      // Voxel/node traversal origin to camera distance in [voxel]
      const float distance = inv_voxel_dim * (camera_position - allocation_origin).norm();

      // Initialise side length in [voxel] of allocated node
      int allocation_size = init_allocation_size;
      int allocation_level = max_level - log2(allocation_size);
      int parent_size = 2 * allocation_size;
      int parent_level = allocation_level - 1;

      Eigen::Vector3f curr_pos_m = allocation_origin;
      Eigen::Vector3f curr_pos_v = inv_voxel_dim * curr_pos_m;
      Eigen::Vector3i curr_node = parent_size * (((curr_pos_v).array().floor()) / parent_size).cast<int>();
      // Fraction of the current position in [voxel] in the current node along the x-, y- and z-axis
      Eigen::Vector3f frac = (curr_pos_v - curr_node.cast<float>()) / parent_size;

      //Current state of T in [voxel]
      Eigen::Vector3f T_max;
      // Increment/Decrement of voxel value along the ray (-1 or +1)
      Eigen::Vector3i step_base;
      // Scaled step_base in [voxel]. Scaling factor will be the current allocation_size
      Eigen::Vector3i step;
      // Travelled distance needed in [voxel] to pass a voxel in x, y and z direction
      Eigen::Vector3f delta_T = parent_size / direction.array().abs(); // [voxel]/[-]

      // Initalize T
      if (direction.x() < 0) {
        step_base.x() = -1;
        T_max.x() = frac.x() * delta_T.x();
      } else {
        step_base.x() = 1;
        T_max.x() = (1 - frac.x()) * delta_T.x();
      }
      if (direction.y() < 0) {
        step_base.y() = -1;
        T_max.y() = frac.y() * delta_T.y();
      } else {
        step_base.y() = 1;
        T_max.y() = (1 - frac.y()) * delta_T.y();
      }
      if (direction.z() < 0) {
        step_base.z() = -1;
        T_max.z() = frac.z() * delta_T.z();
      } else {
        step_base.z() = 1;
        T_max.z() = (1 - frac.z()) * delta_T.z();
      }

      step = parent_size * step_base;

      // Distance travelled in [voxel]
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
            HashType key = oct.hash(curr_node.x(), curr_node.y(), curr_node.z(), parent_level);
            unsigned const idx = parent_count++;
            if (parent_count <= reserved_keys) {
              parent_list[idx] = key;
            }
          } else if (allocation_level >= leaves_level) {
            auto parent_ptr = node_ptr->parent();
            for (int i = 0; i < (1 << NUM_DIM); i++) {
              static_cast<se::VoxelBlock<FieldType> *>(parent_ptr->child(i))->active(true);
            }
          }
        }

        // Update allocation variables
        // Double allocation size every time the allocation distance from the surface is bigger than doubling_ratio * allocation_size
        if ((travelled - inv_voxel_dim * band / 2) > doubling_ratio * allocation_size &&
            (travelled - inv_voxel_dim * band) > 0 &&
            allocation_size < min_allocation_size) {
          allocation_size = 2 * allocation_size;
          parent_size = 2 * allocation_size;

          // Reduce allocation level to coarser level by 1
          allocation_level -= 1;
          parent_level = allocation_level - 1;

          // Update current position along the ray where
          // allocation_origin in [m] and travelled*direction in [voxel]
          curr_pos_v = inv_voxel_dim * allocation_origin + travelled * direction;

          // Re-initialize the curr_node to match the allocation size
          curr_node = parent_size * (((curr_node).array().floor()) / parent_size);

          // Compute fraction of the current position in [voxel] in the updated current node along the x-, y- and z-axis
          frac = (curr_pos_v - curr_node.cast<float>()) / parent_size;

          // Re-initalize delta_T, T_max and step size according to new allocation_size
          delta_T = parent_size / direction.array().abs();
          step = parent_size * step_base;

          if (direction.x() < 0) {
            T_max.x() = travelled + frac.x() * delta_T.x();
          } else {
            T_max.x() = travelled + (1 - frac.x()) * delta_T.x();
          }
          if (direction.y() < 0) {
            T_max.y() = travelled + frac.y() * delta_T.y();
          } else {
            T_max.y() = travelled + (1 - frac.y()) * delta_T.y();
          }
          if (direction.z() < 0) {
            T_max.z() = travelled + frac.z() * delta_T.z();
          } else {
            T_max.z() = travelled + (1 - frac.z()) * delta_T.z();
          }
        }

        // Traverse to closest face crossing of the voxel block/node (i.e. find minimum T_max)
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
  return (size_t) parent_count >= reserved_keys ? reserved_keys : (size_t) parent_count;
}
#endif
