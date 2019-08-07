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

class MultiscaleNodeUpPropagation : public ::testing::Test {
protected:
  virtual void SetUp() {
    size_ = 512;                              // 512 x 512 x 512 voxel^3
    voxel_size_ = 0.005;                      // 5 mm/voxel
    dim_ = size_ * voxel_size_;               // [m^3]
    oct_.init(size_, dim_);

    const int side = se::VoxelBlock<MultiresSDF>::side;
    for(int z = 0; z < size_; z += side) {
      for(int y = 0; y < size_; y += side) {
        for(int x = 0; x < size_; x += side) {
          const Eigen::Vector3i vox(x, y, z);
          alloc_list.push_back(oct_.hash(vox(0), vox(1), vox(2)));
        }
      }
    }
    oct_.allocate(alloc_list.data(), alloc_list.size());
  }

  typedef se::Octree<MultiresSDF> OctreeT;
  OctreeT oct_;
  int size_;
  float voxel_size_;
  float dim_;
  
private:
  std::vector<se::key_t> alloc_list;
};

TEST_F(MultiscaleNodeUpPropagation, Simple) {

      // Change to .ply
//    std::stringstream f;
//    f << "./out/scale_"  + std::to_string(SCALE) + "-sphere-linear_back_move-" + std::to_string(frame) + ".vtk";
//    save3DSlice(oct_,
//                Eigen::Vector3i(0, 0, oct_.size()/2),
//                Eigen::Vector3i(oct_.size(), oct_.size(), oct_.size()/2 + 1),
//                [](const auto& val) { return val.x; }, f.str().c_str());

}