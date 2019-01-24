#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <cassert>
#include <Eigen/StdVector>

namespace se {

  template <typename class_T, typename allocator_T=std::allocator<class_T>>
    class Image {

      public:
        Image(const unsigned w, const unsigned h) : width_(w), height_(h) {
          assert(width_ > 0 && height_ > 0);
          data_.resize(width_ * height_);
        }

        Image(const unsigned w, const unsigned h, const class_T& val) : width_(w), height_(h) {
          assert(width_ > 0 && height_ > 0);
          data_.resize(width_ * height_, val);
        }

        class_T&       operator[](std::size_t idx)       { return data_[idx]; }
        const class_T& operator[](std::size_t idx) const { return data_[idx]; }

        class_T&       operator()(const int x, const int y)       { return data_[x + y*width_]; }
        const class_T& operator()(const int x, const int y) const { return data_[x + y*width_]; }

        std::size_t size()   const   { return width_ * height_; };
        int         width () const { return width_;  };
        int         height() const { return height_; };

        class_T* data()             { return data_.data(); }
        const class_T* data() const { return data_.data(); }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      private:
        const int width_;
        const int height_;
        std::vector<class_T, allocator_T> data_;
    };

} // end namespace se
#endif
