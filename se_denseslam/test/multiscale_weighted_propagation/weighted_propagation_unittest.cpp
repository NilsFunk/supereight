#include <se/octant_ops.hpp>
#include <se/octree.hpp>
#include <se/algorithms/balancing.hpp>
#include <se/functors/axis_aligned_functor.hpp>
#include <se/algorithms/filter.hpp>
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

#define SCALE 3
#define ITERATIONS 50
#define MAX_WEIGHT 5
#define MAX_DIST 3.f

typedef struct ESDF{
  float x;
  float x_last;
  float delta;
  int   y;
  int   delta_y;
} ESDF;

template <>
struct voxel_traits<ESDF> {
  typedef ESDF value_type;
  static inline value_type empty(){ return     {0.f, 0.f, 0.f, 0, 0}; }
  static inline value_type initValue(){ return {1.f, 1.f, 0.f, 0, 0}; }
};

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

          float virt_sample = data.delta*float(data.y) + data.x;
          typedef voxel_traits<T> traits_type;
          typedef typename traits_type::value_type value_type;
          value_type curr_list[8];
          Eigen::Vector3i vox_list[8];
          float sum(0);
          int index(0);

          for (int k = 0; k < stride; k += stride/2)
            for (int j = 0; j < stride; j += stride/2)
              for (int i = 0; i < stride; i += stride/2) {
                const Eigen::Vector3i vox = parent + Eigen::Vector3i(i, j, k);

//                auto curr = block->data(vox, curr_scale - 1);
//
//                // Update SDF value (with 0 <= x_update <= MAX_DIST)
//                curr.x = std::max(std::min(curr.x + data.delta, MAX_DIST), 0.f);
//
//                // Update weight (with 0 <= y <= MAX_WEIGHT)
//                curr.y = std::min(data.delta_y + curr.y, MAX_WEIGHT);
//
//                curr.delta = data.delta;
//                curr.delta_y = data.delta_y;
//                block->data(vox, curr_scale - 1, curr);

                vox_list[index] = parent + Eigen::Vector3i(i, j, k);
                auto curr = block->data(vox_list[index], curr_scale -1);
                curr.delta = virt_sample - curr.x;
                sum += curr.delta;

                curr_list[index] = curr;
                index++;
              }

          for (int i = 0; i < 8; i++) {
            // Update delta_x
            if (sum != 0)
              curr_list[i].delta = data.delta*curr_list[i].delta/sum*8;

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
            if(int(weight - 0.5) == int(weight))
              data.y = ceil(weight);
            else
              data.y = weight;
          } else {
            data.x = 1;
            data.y = 0;
          }

          data.delta = 0;
          data.delta_y = 0;
          block->data(curr, curr_scale + 1, data);
        }
  }
}

template <typename T>
void foreach(const se::MemoryPool<se::VoxelBlock<T> >& block_array) {
  int n = block_array.size();
  for(int i = 0; i < n; ++i) {
    se::VoxelBlock<T>* block = block_array[i];
    const Eigen::Vector3i base = block->coordinates();
    const int side = se::VoxelBlock<T>::side;
    int scale = SCALE;
    float stride = std::max(int(pow(2,scale)),1);
    for(float z = stride/2; z < side; z += stride) {
      for (float y = stride/2; y < side; y += stride) {
        for (float x = stride/2; x < side; x += stride) {
          const Eigen::Vector3f node_w = base.cast<float>() + Eigen::Vector3f(x, y, z);
          auto data = block->data(node_w.cast<int>(), scale);

          const float sample = MAX_DIST;

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

    propagate_down(block, scale);
    propagate_up(block, scale);

  }
  int a = 5;
}

class MultiscaleWeightedPropagationTest : public ::testing::Test {
protected:
  virtual void SetUp() {
    size_ = 16;                              // 512 x 512 x 512 voxel^3
    voxel_size_ = 0.005;                      // 5 mm/voxel
    dim_ = size_ * voxel_size_;         // [m^3]
    oct_.init(size_, dim_);

    const int side = se::VoxelBlock<ESDF>::side;
    for(int z = side/2; z < size_; z += side) {
      for(int y = side/2; y < size_; y += side) {
        for(int x = side/2; x < size_; x += side) {
          const Eigen::Vector3i vox(x, y, z);
          alloc_list.push_back(oct_.hash(vox(0), vox(1), vox(2)));
        }
      }
    }

    oct_.allocate(alloc_list.data(), alloc_list.size());

    const se::MemoryPool<se::VoxelBlock<ESDF> >& block_array = oct_.getBlockBuffer();
    for(unsigned int i = 0; i < block_array.size(); ++i) {
      auto block = block_array[i];
      const Eigen::Vector3i base = block->coordinates();

      for (int z = 0; z < side; z++) {
        for (int y = 0; y < side; y++) {
          for (int x = 0; x < side/2; x++) {
            const Eigen::Vector3f node_w = base.cast<float>() + Eigen::Vector3f(x, y, z);
            auto data = block->data(node_w.cast<int>(), 0);

            // Init voxel values
            data.x = 3;
            data.x_last = 3;
            data.y = 4;

            block->data(node_w.cast<int>(), 0, data);
          }
        }
      }

      for (int z = 0; z < side; z++) {
        for (int y = 0; y < side/4; y++) {
          for (int x = side/2; x < side; x++) {
            const Eigen::Vector3f node_w = base.cast<float>() + Eigen::Vector3f(x, y, z);
            auto data = block->data(node_w.cast<int>(), 0);

            // Init voxel values
            data.x = 3;
            data.y = 4;

            block->data(node_w.cast<int>(), 0, data);
          }
        }
      }

      propagate_up(block, 0);

    }


  }

  typedef se::Octree<ESDF> OctreeT;
  OctreeT oct_;
  int size_;
  float voxel_size_;
  float dim_;

private:
  std::vector<se::key_t> alloc_list;
};

TEST_F(MultiscaleWeightedPropagationTest, WeightedPropagation) {
  int frames = ITERATIONS + 1;
  for (int frame = 0; frame <= frames; frame++) {
    const se::MemoryPool<se::VoxelBlock<ESDF> >& block_array = oct_.getBlockBuffer();

    std::stringstream f;

    f << "/home/nils/workspace_ptp/catkin_ws/src/probabilistic_trajectory_planning_ros/ext/probabilistic_trajectory_planning/src/ext/supereight/se_denseslam/test/out/scale_" + std::to_string(SCALE) + "_block_integration_boundary_1-" + std::to_string(frame) + ".vtk";

    save3DSlice(oct_,
                Eigen::Vector3i(0, 0, oct_.size()/2),
                Eigen::Vector3i(oct_.size(), oct_.size(), oct_.size()/2 + 1),
                [](const auto& val) { return val.x; }, f.str().c_str());
    foreach(block_array);
  }
}

