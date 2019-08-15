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
 * */

#ifndef MULTIRES_BFUSION_MAPPING_HPP
#define MULTIRES_BFUSION_MAPPING_HPP

#include <se/node.hpp>
#include <se/functors/projective_functor.hpp>
#include <se/constant_parameters.h>
#include <se/image/image.hpp>
#include "bspline_lookup.cc"
#include <se/octree.hpp>
#include <se/volume_traits.hpp>
#include <se/algorithms/filter.hpp>
#include <se/functors/for_each.hpp>

namespace se {
  namespace multires {
    namespace ofusion {

      /**
       * Perform bilinear interpolation on a depth image. See
       * https://en.wikipedia.org/wiki/Bilinear_interpolation for more details.
       *
       * \param[in] depth The depth image to interpolate.
       * \param[in] proj The coordinates on the image at which the interpolation
       * should be computed.
       * \return The value of the interpolated depth at proj.
       */
      float interpDepth(const se::Image<float>& depth, const Eigen::Vector2f& proj) {
        // https://en.wikipedia.org/wiki/Bilinear_interpolation

        // Pixels version
        const float x1 = (floorf(proj.x()));
        const float y1 = (floorf(proj.y() + 1));
        const float x2 = (floorf(proj.x() + 1));
        const float y2 = (floorf(proj.y()));

        const float d11 = depth(int(x1), int(y1));
        const float d12 = depth(int(x1), int(y2));
        const float d21 = depth(int(x2), int(y1));
        const float d22 = depth(int(x2), int(y2));

        if ( d11 == 0.f || d12 == 0.f || d21 == 0.f || d22 == 0.f )
          return 0.f;

        const float f11 = 1.f / d11;
        const float f12 = 1.f / d12;
        const float f21 = 1.f / d21;
        const float f22 = 1.f / d22;

        // Filtering version
        const float d =  1.f /
                         ( (   f11 * (x2 - proj.x()) * (y2 - proj.y())
                               + f21 * (proj.x() - x1) * (y2 - proj.y())
                               + f12 * (x2 - proj.x()) * (proj.y() - y1)
                               + f22 * (proj.x() - x1) * (proj.y() - y1)
                           ) / ((x2 - x1) * (y2 - y1))
                         );

        static const float interp_thresh = 0.05f;
        if (fabs(d - d11) < interp_thresh
            && fabs(d - d12) < interp_thresh
            && fabs(d - d21) < interp_thresh
            && fabs(d - d22) < interp_thresh) {
          return d;
        } else {
          return depth(int(proj.x() + 0.5f), int(proj.y() + 0.5f));
        }
      }
      /**
       * Compute the value of the q_cdf spline using a lookup table. This implements
       * equation (7) from \cite VespaRAL18.
       *
       * \param[in] t Where to compute the value of the spline at.
       * \return The value of the spline.
       */
      static inline float bspline_memoized(float t) {
        float value = 0.f;
        constexpr float inverseRange = 1/6.f;
        if (t >= -3.0f && t <= 3.0f) {
          unsigned int idx = ((t + 3.f)*inverseRange)*(bspline_num_samples - 1) + 0.5f;
          return bspline_lookup[idx];
        } else if(t > 3) {
          value = 1.f;
        }
        return value;
      }

      /**
       * Compute the occupancy probability along the ray from the camera. This
       * implements equation (6) from \cite VespaRAL18.
       *
       * \param[in] val The point on the ray at which the occupancy probability is
       * computed. The point is expressed using the ray parametric equation.
       * \param[in]
       * \return The occupancy probability.
       */
      static inline float H(const float val, const float) {
        const float Q_1 = bspline_memoized(val);
        const float Q_2 = bspline_memoized(val - 3);
        return Q_1 - Q_2 * 0.5f;
      }

      /**
       * Perform a log-odds update of the occupancy probability. This implements
       * equations (8) and (9) from \cite VespaRAL18.
       */
      static inline float updateLogs(const float prior, const float sample) {
        return (prior + log2(sample / (1.f - sample)));
      }

      /**
       * Weight the occupancy by the time since the last update, acting as a
       * forgetting factor. This implements equation (10) from \cite VespaRAL18.
       */
      static inline float applyWindow(const float occupancy,
                                      const float,
                                      const float delta_t,
                                      const float tau) {
        float fraction = 1.f / (1.f + (delta_t / tau));
        fraction = std::max(0.5f, fraction);
        return occupancy * fraction;
      }

      /**
       * Update the maximum occupuancy of a voxel block starting scale 0 to voxel block scale
       *
       * \param[in] block VoxelBlock to be updated
      */
      template <typename T>
      void propagate_up(se::VoxelBlock<T>* block, const int scale) {
        const Eigen::Vector3i base = block->coordinates();
        const int side = se::VoxelBlock<T>::side;
        for(int curr_scale = scale; curr_scale < se::math::log2_const(side); ++curr_scale) {
          const int stride = 1 << (curr_scale + 1);
          for(int z = 0; z < side; z += stride)
            for(int y = 0; y < side; y += stride)
              for(int x = 0; x < side; x += stride) {
                const Eigen::Vector3i curr = base + Eigen::Vector3i(x, y, z);
                float mean = 0;
                float x_max = -TOP_CLAMP;
                for(int k = 0; k < stride; k += stride/2)
                  for(int j = 0; j < stride; j += stride/2 )
                    for(int i = 0; i < stride; i += stride/2) {
                      auto tmp = block->data(curr + Eigen::Vector3i(i, j , k), curr_scale);
                      mean += tmp.x;
                      if (tmp.x_max > x_max)
                        x_max = tmp.x_max;
                    }
                auto data = block->data(curr, curr_scale + 1);
                data.x = mean / 8;
                data.x_max = x_max;
                block->data(curr, curr_scale + 1, data);
              }
        }
      }

      template <typename T>
      void propagate_up(se::Node<T>* node, const int max_level,
                        const unsigned timestamp) {
        node->timestamp(timestamp);

        if(!node->parent()) {
          return;
        }

        float x_max = BOTTOM_CLAMP;
        for(int i = 0; i < 8; ++i) {
          const auto& tmp = node->value_[i];
          if (tmp.x_max > x_max)
            x_max = tmp.x_max;
        }

        const unsigned int id = se::child_id(node->code_,
                                             se::keyops::level(node->code_), max_level);
        auto& data = node->parent()->value_[id];
        data.x = x_max;
        data.x_max = x_max;
      }

      struct multires_block_update {
        multires_block_update(
            const se::Octree<MultiresOFusion>& octree,
            const Sophus::SE3f& T,
            const Eigen::Matrix4f& calib,
            const float vsize,
            const Eigen::Vector3f& off,
            const se::Image<float>& d,
            const int f,
            const float m) :
            map(octree),
            Tcw(T),
            K(calib),
            voxel_size(vsize),
            offset(off),
            depth(d),
            frame(f),
            mu(m) {}

        const se::Octree<MultiresOFusion>& map;
        const Sophus::SE3f& Tcw;
        const Eigen::Matrix4f& K;
        float voxel_size;
        const Eigen::Vector3f& offset;
        const se::Image<float>& depth;
        const int frame;
        float mu;

        void operator()(se::VoxelBlock<MultiresOFusion>* block) {
          constexpr int side = se::VoxelBlock<MultiresOFusion>::side;
          const Eigen::Vector3i base = block->coordinates();
          const int scale = 0;
          const int stride = 1 << scale;
          bool visible = false;

          const Eigen::Vector3f delta = Tcw.rotationMatrix() * Eigen::Vector3f(voxel_size, 0, 0);
          const Eigen::Vector3f cameraDelta = K.topLeftCorner<3, 3>() * delta;
          for (int z = 0; z < side; z += stride) {
            for (int y = 0; y < side; y += stride) {
              Eigen::Vector3i pix = base + Eigen::Vector3i(0, y, z);
              Eigen::Vector3f start = Tcw * (voxel_size * (pix.cast<float>() + stride * offset));
              Eigen::Vector3f camerastart = K.topLeftCorner<3, 3>() * start;
              for (int x = 0; x < side; x += stride, pix.x() += stride) {
                const Eigen::Vector3f camera_voxel = camerastart + (x * cameraDelta);
                const Eigen::Vector3f pos = start + (x * delta);
                if (pos.z() < 0.0001f) continue;

                const float inverse_depth = 1.f / camera_voxel.z();
                const Eigen::Vector2f pixel = Eigen::Vector2f(
                    camera_voxel.x() * inverse_depth + 0.5f,
                    camera_voxel.y() * inverse_depth + 0.5f);
                if (pixel.x() < 0.5f || pixel.x() > depth.width() - 1.5f ||
                    pixel.y() < 0.5f || pixel.y() > depth.height() - 1.5f)
                  continue;
                visible = true;
                const Eigen::Vector2i px = pixel.cast<int>();
                const float depthSample = depth[px.x() + depth.width() * px.y()];
                // continue on invalid depth measurement
                if (depthSample <= 0) continue;

                // Compute the occupancy probability for the current measurement.
                const float diff = (pos.z() - depthSample)
                                   * std::sqrt( 1 + se::math::sq(pos.x() / pos.z()) + se::math::sq(pos.y() / pos.z()));
                float sigma = se::math::clamp(mu * se::math::sq(pos.z()),
                                              2 * voxel_size, 0.05f);
                float sample = H(diff/sigma, pos.z());
                if (sample == 0.5f)
                  continue;
                sample = se::math::clamp(sample, 0.03f, 0.97f);

                auto data = block->data(pix, scale);

                // Update the occupancy probability
                const double delta_t = (double)(frame - data.y) / 30;
//                data.x = applyWindow(data.x, SURF_BOUNDARY, delta_t, CAPITAL_T);
                data.x = se::math::clamp(updateLogs(data.x, sample), BOTTOM_CLAMP, TOP_CLAMP);
                data.x_max = data.x;
                data.y = frame;

                block->data(pix, scale, data);
              }
            }
          }
          propagate_up(block, scale);
          block->active(visible);
        }
      };

      template <typename T>
      void integrate(se::Octree<T>& , const Sophus::SE3f& , const
      Eigen::Matrix4f& , float , const Eigen::Vector3f& , const
                     se::Image<float>& , float , const unsigned) {
      }

      template <>void integrate(se::Octree<MultiresOFusion>& map, const Sophus::SE3f& Tcw, const
      Eigen::Matrix4f& K, float voxelsize, const Eigen::Vector3f& offset, const
                                se::Image<float>& depth, float mu, const unsigned frame) {
        // Filter visible blocks
        using namespace std::placeholders;
        std::vector<se::VoxelBlock<MultiresOFusion>*> active_list;
        auto& block_array = map.getBlockBuffer();
        auto is_active_predicate = [](const se::VoxelBlock<MultiresOFusion>* b) {
          return b->active();
        };
        const Eigen::Vector2i framesize(depth.width(), depth.height());
        const Eigen::Matrix4f Pcw = K*Tcw.matrix();
        auto in_frustum_predicate =
            std::bind(se::algorithms::in_frustum<se::VoxelBlock<MultiresOFusion>>, _1,
                      voxelsize, Pcw, framesize);
        se::algorithms::filter(active_list, block_array, is_active_predicate,
                               in_frustum_predicate);

        struct multires_block_update funct(map, Tcw, K, voxelsize,
                                           offset, depth, frame, mu);
        se::functor::internal::parallel_for_each(active_list, funct);

        std::deque<Node<MultiresOFusion>*> prop_list;
        std::mutex deque_mutex;

        std::vector<se::Node<MultiresOFusion>*> active_node_list;
        auto& nodes_array = map.getNodesBuffer();
        auto is_active_node_predicate = [](const se::Node<MultiresOFusion>* n) {
          return n->active();
        };
        algorithms::filter(active_node_list, nodes_array, is_active_node_predicate);

        for(const auto& n : active_node_list) {
          for(int i = 0; i < 8; ++i) {
            auto& data = n->value_[i];
            data.x += -5.015;
            data.x = std::max(data.x, voxel_traits<MultiresOFusion>::freeThresh());
            data.x_max = data.x;
            data.y = frame;
          }
        }

//        for(const auto& n : active_node_list) {
//          for(int i = 0; i < 8; ++i) {
//            if (n->child(i) == NULL) {
//              auto& data = n->value_[i];
//              data.x += -5.015;
//              data.x = std::max(data.x, voxel_traits<MultiresOFusion>::freeThresh());
//              data.x_max = data.x;
//              data.y = frame;
//            } else if (!n->child(i)->isLeaf()){
//              active_node_list.push_back(n->child(i));
//              n->child(i)->active(true);
//            }
//          }
//////          if(n->parent() && n->children_mask_ == 0) {
////          if(n->parent()) {
////            prop_list.push_back(n->parent());
////          }
////          n->active(false);
//        }

        for(const auto& b : active_list) {
          if(b->parent()) {
            prop_list.push_back(b->parent());
            const unsigned int id = se::child_id(b->code_,
                                                 se::keyops::level(b->code_), map.max_level());
            auto data = b->data(b->coordinates(), se::math::log2_const(se::VoxelBlock<MultiresOFusion>::side));
            auto& parent_data = b->parent()->value_[id];
            parent_data = data;
            parent_data.x = parent_data.x_max;
          }
        }

        while(!prop_list.empty()) {
          Node<MultiresOFusion>* n = prop_list.front();
          prop_list.pop_front();
          if(n->timestamp() == frame) continue;
          propagate_up(n, map.max_level(), frame);
          if(n->parent()) prop_list.push_back(n->parent());
        }

        for(const auto& n : active_node_list) {
          prop_list.push_back(n);
          n->active(false);
        }

        while(!prop_list.empty()) {
          Node<MultiresOFusion>* n = prop_list.front();
          prop_list.pop_front();
          propagate_up(n, map.max_level(), frame);
          if(n->parent()) prop_list.push_back(n->parent());
        }

        int count = 0;
        int num_blocks = block_array.size();
        for (int i = 0; i < num_blocks; i++) {
          auto b = block_array[count];
          const unsigned int id = se::child_id(b->code_,
                                               se::keyops::level(b->code_), map.max_level());
          if (b->parent()) {
            if (b->parent()->value_[id].x == voxel_traits<MultiresOFusion>::freeThresh()) {
              b->parent()->child(id) = NULL;
              block_array.erase(count);
              continue;
            }
          }
          count++;
        }

        count = 0;
        int num_nodes = nodes_array.size();
        for (int i = 1; i < num_nodes; i++) {
          auto n = nodes_array[count];
          const unsigned int id = se::child_id(n->code_,
                                               se::keyops::level(n->code_), map.max_level());
          if (n->parent()) {
            if (n->parent()->value_[id].x == voxel_traits<MultiresOFusion>::freeThresh()) {
              n->parent()->child(id) = NULL;
              nodes_array.erase(count);
              continue;
            }
          } else if (n->side_ != map.size()) {
            n->parent()->child(id) = NULL;
            nodes_array.erase(count);
            continue;
          }
          count++;
        }

      }
    }
  }
}
#endif
