/*

Copyright 2016 Emanuele Vespa, Imperial College London

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef MEM_POOL_H
#define MEM_POOL_H

#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>

namespace se {
/*! \brief Manage the memory allocated for Octree nodes.
 *
 * \note The memory is managed using a vector of shared pointers. This allows
 * safely erasing elements.
 */
template <typename BlockType>
  class MemoryPool2 {
    public:
      MemoryPool2(){
        current_block_ = 0;
        reserved_ = 0;
      }

      ~MemoryPool2(){
      }

      /*! \brief The number of BlockType elements stored in the MemoryPool2.
       */
      size_t size() const { return current_block_; };

      /*! \brief Return a pointer to the BlockType element with index i.
       */
      BlockType* operator[](const size_t i) const {
        return data_[i].get();
      }

      /*! \brief Reserve memory for an additional n BlockType elements.
       */
      void reserve(const size_t n) {
        bool requires_realloc = (current_block_ + n) > reserved_;
        if (requires_realloc)
          expand(n);
      }

      /*! \brief Add a BlockType element to the MemoryPool2 and return a pointer
       * to it.
       *
       * \note This function should be thread-safe.
       *
       * \warning No bounds checking is performed. The MemoryPool2 must have
       * enough memory allocated using MemoryPool2::reserve to accomodate the
       * new element.
       */
      BlockType * acquire_block(){
        // Fetch-add returns the value before increment
        int current = current_block_.fetch_add(1);
        BlockType * ptr = data_[current].get();
        return ptr;
      }

      /*! \brief Erase the BlockType element at index i.
       *
       * \note This function should be thread-safe.
       *
       * \warning No bounds checking is performed. Ensure i is smaller than
       * MemoryPool::size.
       */
      void erase(const size_t i) {
        // Create a mutex local to the function. The unique lock is an object
        // that guarantees the mutex will be unlocked on destruction (e.g. when
        // the function exits or after an exception is raised).
        // This code should be thread safe in C++11.
        // https://stackoverflow.com/questions/14106653/are-function-local-static-mutexes-thread-safe
        static std::mutex mtx;
        std::unique_lock<std::mutex> lock(mtx);

        // Decrement the number of elements and erase the requested element.
        current_block_.fetch_sub(1);
        data_.erase(data_.begin() + i);
      }

    private:
      size_t reserved_;
      std::atomic<unsigned int> current_block_;
      const int pagesize_ = 1024; // # of blocks per page
      std::vector<std::shared_ptr<BlockType> > data_;

      void expand(const size_t n) {

        // std::cout << "Allocating " << n << " blocks" << std::endl;
        for (size_t p = 0; p < n; ++p) {
          data_.push_back(std::make_shared<BlockType>());
          reserved_++;
        }
        // std::cout << "Reserved " << reserved_ << " blocks" << std::endl;
      }

      // Disabling copy-constructor
      MemoryPool2(const MemoryPool2& m);
  };
}
#endif
