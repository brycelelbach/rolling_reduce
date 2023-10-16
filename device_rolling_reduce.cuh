/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.
 * Copyright (c) 2011-2023, NVIDIA CORPORATION.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

/**
 * @file cub::DeviceRollingReduce provides device-wide, parallel operations for
 *       computing a rolling reduction across a sequence of data items residing
 *       within device-accessible memory.
 */

#pragma once

#include <cub/config.cuh>

#include "dispatch_rolling_reduce.cuh"
#include <cub/thread/thread_operators.cuh>
#include <cub/util_deprecated.cuh>

CUB_NAMESPACE_BEGIN

struct DeviceRollingReduce
{
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ReductionOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  RollingReduce(void *d_temp_storage,
                size_t &temp_storage_bytes,
                InputIteratorT d_in,
                OutputIteratorT d_out,
                ReductionOpT reduction_op,
                int num_items,
                int window_size,
                cudaStream_t stream = 0)
  {
      // Signed integer type for global offsets
      using OffsetT = int;

      return DispatchRollingReduce<InputIteratorT,
                                   OutputIteratorT,
                                   ReductionOpT,
                                   OffsetT>::Dispatch(d_temp_storage,
                                                      temp_storage_bytes,
                                                      d_in,
                                                      d_out,
                                                      reduction_op,
                                                      num_items,
                                                      window_size,
                                                      stream);
  }

  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ReductionOpT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION static cudaError_t
  RollingReduce(void *d_temp_storage,
                size_t &temp_storage_bytes,
                InputIteratorT d_in,
                OutputIteratorT d_out,
                ReductionOpT reduction_op,
                int num_items,
                int window_size,
                cudaStream_t stream,
                bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return RollingReduce<InputIteratorT,
                         OutputIteratorT,
                         ReductionOpT>(d_temp_storage,
                                       temp_storage_bytes,
                                       d_in,
                                       d_out,
                                       reduction_op,
                                       num_items,
                                       window_size,
                                       stream);
  }

};

CUB_NAMESPACE_END


