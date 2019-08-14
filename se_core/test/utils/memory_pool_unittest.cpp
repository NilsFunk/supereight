/*
 * Copyright 2019 Sotiris Papatheodorou, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */


#include "utils/memory_pool.hpp"
#include "gtest/gtest.h"



typedef float testT;



// Create a MemoryPool and set the values of some of its elements.
class MemoryPoolTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      // Reserve memory for the pool.
      pool_.reserve(num_elements_);

      // Set the value of some elements.
      for (size_t i = 0; i < num_elements_; ++i) {
        testT* e = pool_.acquire_block();
        *e = i * value_increment_;
      }
    }

    se::MemoryPool<testT> pool_;
    const float value_increment_ = 1.f;
    // The page size is currently hardcoded to 1024 elements.
    const size_t num_elements_ = 1026;
};





// Test that the MemoryPool contains the expected number of elements.
TEST_F(MemoryPoolTest, Init) {
  EXPECT_EQ(pool_.size(), num_elements_);

  for (size_t i = 0; i < pool_.size(); ++i) {
    testT* e = pool_[i];
    EXPECT_EQ(*e, i * value_increment_);
  }
}

