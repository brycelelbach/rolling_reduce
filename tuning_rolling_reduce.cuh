/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

CUB_NAMESPACE_BEGIN

/**
 * Parameterizable tuning policy type for AgentRollingReduce
 *
 * @tparam DelayConstructorT
 *   Implementation detail, do not specify directly, requirements on the
 *   content of this type are subject to breaking change.
 */
template <int _BLOCK_THREADS,
          int _ITEMS_PER_THREAD                = 1,
          BlockLoadAlgorithm _LOAD_ALGORITHM   = BLOCK_LOAD_DIRECT,
          CacheLoadModifier _LOAD_MODIFIER     = LOAD_DEFAULT,
          BlockScanAlgorithm _SCAN_ALGORITHM   = BLOCK_SCAN_WARP_SCANS,
          BlockStoreAlgorithm _STORE_ALGORITHM = BLOCK_STORE_DIRECT,
          typename DelayConstructorT           = detail::fixed_delay_constructor_t<350, 450>>
struct AgentRollingReducePolicy
{
  static constexpr int BLOCK_THREADS    = _BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;

  static constexpr BlockLoadAlgorithm LOAD_ALGORITHM   = _LOAD_ALGORITHM;
  static constexpr CacheLoadModifier LOAD_MODIFIER     = _LOAD_MODIFIER;
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM   = _SCAN_ALGORITHM;
  static constexpr BlockStoreAlgorithm STORE_ALGORITHM = _STORE_ALGORITHM;

  struct detail
  {
    using delay_constructor_t = DelayConstructorT;
  };
};

template <typename KeysInputIteratorT,
          typename AccumT,
          typename ValueT = AccumT,
          typename ScanOpT = Sum>
struct DeviceRollingReduceManualPolicy
{
  struct ManualTuning : ChainedPolicy<800, ManualTuning, ManualTuning>
  {
    using ScanByKeyPolicyT =
      AgentRollingReducePolicy<512,
                               6,
                               BLOCK_LOAD_WARP_TRANSPOSE,
                               LOAD_DEFAULT,
                               BLOCK_SCAN_WARP_SCANS,
                               BLOCK_STORE_WARP_TRANSPOSE,
                               detail::default_reduce_by_key_delay_constructor_t<AccumT, int>>;
  };

  using MaxPolicy = ManualTuning;
};

CUB_NAMESPACE_END
