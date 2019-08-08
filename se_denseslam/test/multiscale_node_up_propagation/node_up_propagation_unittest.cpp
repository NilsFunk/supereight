#include <se/octree.hpp>
#include <gtest/gtest.h>
#include "../src/multires/mapping_impl.hpp"

typedef struct testT {
  float x;
  float x_last;
  int y;
  int delta_y;
} testT;

template <>
struct voxel_traits<testT> {
  typedef testT value_type;
  static inline value_type empty(){ return {0.f, 0.f, 1, 0};}
  static inline value_type initValue(){ return {0.f, 0.f, 1, 0};}
};


class MultiscaleNodeUpPropagation : public ::testing::Test {
protected:
  virtual void SetUp() {
    size_ = 64;                               // 512 x 512 x 512 voxel^3
    max_level_ = se::math::log2_const(size_);
    voxel_size_ = 0.005;                      // 5 mm/voxel
    dim_ = size_ * voxel_size_;               // [m^3]
    oct_.init(size_, dim_);

    side_ = se::VoxelBlock<testT>::side;
    for(int z = 0; z < size_; z += side_) {
      for(int y = 0; y < size_; y += side_) {
        for(int x = 0; x < size_; x += side_) {
          const Eigen::Vector3i vox(x, y, z);
          alloc_list.push_back(oct_.hash(vox(0), vox(1), vox(2)));
        }
      }
    }
    oct_.allocate(alloc_list.data(), alloc_list.size());
  }

  typedef se::Octree<testT> OctreeT;
  OctreeT oct_;
  int size_;
  int side_;
  int max_level_;
  float voxel_size_;
  float dim_;
  
private:
  std::vector<se::key_t> alloc_list;
};

TEST_F(MultiscaleNodeUpPropagation, Simple) {
  std::vector<se::VoxelBlock<testT>*> active_list;
  std::deque<se::Node<testT>*> prop_list;

  // Update some voxels.
  constexpr size_t num_voxels = 4;
  const Eigen::Vector3i voxels[num_voxels] =
      {{0, 0, 0}, {8, 8, 0}, {48, 48, 0}, {56, 56, 0}};

  for (size_t i = 0; i < num_voxels; ++i) {
    for (int x = 0; x < side_; x++) {
      for (int y = 0; y < side_; y++) {
        for (int z = 0; z < side_; z++) {
          Eigen::Vector3i voxel_tmp = voxels[i] + Eigen::Vector3i(x, y, z);
          oct_.set(voxel_tmp.x(), voxel_tmp.y(), voxel_tmp.z(), {1, 0, 1, 0});
        }
      }
    }
    se::VoxelBlock<testT>* vb = oct_.fetch(voxels[i].x(), voxels[i].y(), voxels[i].z());
    active_list.push_back(vb);
  }

  for (const auto& b : active_list) {
    se::multires::propagate_up(b, 0);
  }


  for(const auto& b : active_list) {
    if(b->parent()) {
      prop_list.push_back(b->parent());

      const unsigned int id = se::child_id(b->code_,
                                           se::keyops::level(b->code_),  max_level_);
      auto data = b->data(b->coordinates(), se::math::log2_const(se::VoxelBlock<MultiresSDF>::side));
      auto& parent_data = b->parent()->value_[id];
      parent_data = data;
    }
  }

  int frame = 1;

  while(!prop_list.empty()) {
    se::Node<testT>* n = prop_list.front();
    prop_list.pop_front();
    if(n->timestamp() == frame) continue;
    se::multires::propagate_up(n, max_level_, frame);
    if(n->parent()) prop_list.push_back(n->parent());
  }

  se::Node<testT>* n = oct_.root();
  ASSERT_EQ(n->value_[0].x, 2.f/64);
  ASSERT_EQ(n->value_[1].x, 0);
  ASSERT_EQ(n->value_[2].x, 0);
  ASSERT_EQ(n->value_[3].x, 2.f/64);
  ASSERT_EQ(n->value_[5].x, 0);
  ASSERT_EQ(n->value_[6].x, 0);
  ASSERT_EQ(n->value_[7].x, 0);

  n = n->child(0);
  ASSERT_EQ(n->value_[0].x, 2.f/8);
  ASSERT_EQ(n->value_[1].x, 0);
  ASSERT_EQ(n->value_[2].x, 0);
  ASSERT_EQ(n->value_[3].x, 0);
  ASSERT_EQ(n->value_[5].x, 0);
  ASSERT_EQ(n->value_[6].x, 0);
  ASSERT_EQ(n->value_[7].x, 0);

  n = n->parent()->child(3);
  ASSERT_EQ(n->value_[0].x, 0);
  ASSERT_EQ(n->value_[1].x, 0);
  ASSERT_EQ(n->value_[2].x, 0);
  ASSERT_EQ(n->value_[3].x, 2.f/8);
  ASSERT_EQ(n->value_[5].x, 0);
  ASSERT_EQ(n->value_[6].x, 0);
  ASSERT_EQ(n->value_[7].x, 0);
}