/******************************************************************************
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
 * @file DeviceScan provides device-wide, parallel operations for computing a
 *       prefix scan across a sequence of data items residing within
 *       device-accessible memory.
 */

#pragma once

#include <cub/config.cuh>
#include "agent_rolling_reduce.cuh"
#include <cub/device/dispatch/dispatch_scan_by_key.cuh>
#include <cub/device/dispatch/tuning/tuning_scan_by_key.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_macro.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <iterator>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

/**
 * @brief Rolling reduce kernel entry point (multi-block)
 */
template <typename ChainedPolicyT,
          typename KeysInputIteratorT,
          typename UnzippedInputIteratorT,
          typename ZippedInputIteratorT,
          typename OutputIteratorT,
          typename RollingReduceTileStateT,
          typename UnzippedReductionOpT,
          typename OffsetT,
          typename UnzippedAccumT,
          typename KeyT = cub::detail::value_t<KeysInputIteratorT>>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::ScanByKeyPolicyT::BLOCK_THREADS))
CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceRollingReduceKernel(KeysInputIteratorT d_keys_in,
                                                            KeyT *d_keys_prev_in,
                                                            UnzippedInputIteratorT d_in,
                                                            ZippedInputIteratorT d_zipped_in,
                                                            OutputIteratorT d_out,
                                                            RollingReduceTileStateT tile_state,
                                                            int start_tile,
                                                            UnzippedReductionOpT reduction_op,
                                                            OffsetT num_items)
{
  using RollingReducePolicyT =
    typename ChainedPolicyT::ActivePolicy::ScanByKeyPolicyT;

  // Thread block type for scanning input tiles
  using AgentRollingReduceT = AgentRollingReduce<RollingReducePolicyT,
                                                 UnzippedInputIteratorT,
                                                 OutputIteratorT,
                                                 UnzippedReductionOpT,
                                                 OffsetT,
                                                 UnzippedAccumT>;

  // Shared memory for AgentRollingReduce
  __shared__ typename AgentRollingReduceT::TempStorage temp_storage;

  // Process tiles
  AgentRollingReduceT(temp_storage,
                      d_keys_in,
                      d_keys_prev_in,
                      d_in,
                      d_zipped_in,
                      d_out,
                      reduction_op)
    .ConsumeRange(num_items, tile_state, start_tile);
}

/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels
 *        for DeviceRollingReduce
 *
 */
template <
  typename UnzippedInputIteratorT,
  typename OutputIteratorT,
  typename UnzippedReductionOpT,
  typename OffsetT,
  typename UnzippedAccumT =
    detail::accumulator_t<UnzippedReductionOpT,
                          cub::detail::value_t<UnzippedInputIteratorT>,
                          cub::detail::value_t<UnzippedInputIteratorT>>,
  typename SelectedPolicy = DeviceScanByKeyPolicy<
    thrust::transform_iterator<IndexToWindow<OffsetT>, thrust::counting_iterator<OffsetT>>,
    thrust::tuple<UnzippedAccumT, UnzippedAccumT>,
    cub::detail::value_t<thrust::zip_iterator<
      thrust::tuple<UnzippedInputIteratorT, thrust::reverse_iterator<UnzippedInputIteratorT>>>>,
    ZippedReduce<UnzippedReductionOpT>>>
struct DispatchRollingReduce : SelectedPolicy
{
  //---------------------------------------------------------------------
  // Constants and Types
  //---------------------------------------------------------------------

  static constexpr int INIT_KERNEL_THREADS = 128;

  using KeysInputIteratorT   = thrust::transform_iterator<
    IndexToWindow<OffsetT>, thrust::counting_iterator<OffsetT>>;
  using ZippedInputIteratorT = thrust::zip_iterator<
    thrust::tuple<UnzippedInputIteratorT, thrust::reverse_iterator<UnzippedInputIteratorT>>>;

  using AccumT = thrust::tuple<UnzippedAccumT, UnzippedAccumT>;

  using KeyT           = OffsetT;
  using InputT         = cub::detail::value_t<ZippedInputIteratorT>;
  using UnzippedInputT = cub::detail::value_t<UnzippedInputIteratorT>;

  /// Device-accessible allocation of temporary storage. When `nullptr`, the
  /// required allocation size is written to `temp_storage_bytes` and no work
  /// is done.
  void *d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t &temp_storage_bytes;

  /// Iterator to the input sequence of key items
  KeysInputIteratorT d_keys_in;

  /// Iterator to the input sequence of value items
  UnzippedInputIteratorT d_in;

  /// Iterator to the input sequence of value items
  ZippedInputIteratorT d_zipped_in;

  /// Iterator to the input sequence of value items
  OutputIteratorT d_out;

  /// Binary scan functor
  UnzippedReductionOpT reduction_op;

  /// Total number of input items (i.e., the length of `d_in`)
  OffsetT num_items;

  /// Rolling window size
  OffsetT window_size;

  /// CUDA stream to launch kernels within.
  cudaStream_t stream;
  int ptx_version;

  CUB_RUNTIME_FUNCTION CUB_FORCE_INLINE
  DispatchRollingReduce(void *d_temp_storage,
                    size_t &temp_storage_bytes,
                    UnzippedInputIteratorT d_in,
                    OutputIteratorT d_out,
                    UnzippedReductionOpT reduction_op,
                    OffsetT num_items,
                    OffsetT window_size,
                    cudaStream_t stream,
                    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys_in(thrust::make_transform_iterator(
          thrust::counting_iterator<OffsetT>(0), IndexToWindow(window_size)) + window_size - 1)
      , d_in(d_in)
      , d_zipped_in(thrust::make_zip_iterator(thrust::make_tuple(
          d_in, thrust::make_reverse_iterator(d_in + num_items))) + window_size - 1)
      , d_out(d_out)
      , reduction_op(reduction_op)
      , num_items(num_items - window_size + 1)
      , window_size(window_size)
      , stream(stream)
      , ptx_version(ptx_version)
  {}

  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION CUB_FORCE_INLINE
  DispatchRollingReduce(void *d_temp_storage,
                    size_t &temp_storage_bytes,
                    UnzippedInputIteratorT d_in,
                    OutputIteratorT d_out,
                    UnzippedReductionOpT reduction_op,
                    OffsetT num_items,
                    OffsetT window_size,
                    cudaStream_t stream,
                    bool debug_synchronous,
                    int ptx_version)
      : DispatchRollingReduce(d_temp_storage, temp_storage_bytes, d_in, d_out,
          reduction_op, num_items, window_size, stream, ptx_version)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG
  }

  template <typename ActivePolicyT, typename InitKernel, typename ScanKernel>
  CUB_RUNTIME_FUNCTION __host__ CUB_FORCE_INLINE cudaError_t
  Invoke(InitKernel init_kernel, ScanKernel scan_kernel)
  {
    using Policy = typename ActivePolicyT::ScanByKeyPolicyT;
    using RollingReduceTileStateT = ReduceByKeyScanTileState<AccumT, OffsetT>;

    cudaError error = cudaSuccess;
    do
    {
      // Get device ordinal
      int device_ordinal;
      error = CubDebug(cudaGetDevice(&device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Number of input tiles
      int tile_size = Policy::BLOCK_THREADS * Policy::ITEMS_PER_THREAD;
      int num_tiles =
        static_cast<int>(cub::DivideAndRoundUp(num_items, tile_size));

      // Specify temporary storage allocation requirements
      size_t allocation_sizes[2];
      error = CubDebug(RollingReduceTileStateT::AllocationSize(num_tiles, allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        break; // bytes needed for tile status descriptors
      }

      allocation_sizes[1] = sizeof(KeyT) * (num_tiles + 1);

      // Compute allocation pointers into the single storage blob (or compute
      // the necessary size of the blob)
      void *allocations[2] = {};

      error = CubDebug(
        AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        break;
      }

      if (d_temp_storage == NULL)
      {
        // Return if the caller is simply requesting the size of the storage
        // allocation
        break;
      }

      // Return if empty problem
      if (num_items == 0)
      {
        break;
      }

      KeyT *d_keys_prev_in = reinterpret_cast<KeyT *>(allocations[1]);

      // Construct the tile status interface
      RollingReduceTileStateT tile_state;
      error = CubDebug(tile_state.Init(num_tiles, allocations[0], allocation_sizes[0]));
      if (cudaSuccess != error)
      {
        break;
      }

      // Log init_kernel configuration
      int init_grid_size = cub::DivideAndRoundUp(num_tiles,
                                                 INIT_KERNEL_THREADS);
      #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      _CubLog("Invoking DispatchRollingReduce init_kernel<<<%d, %d, 0, %lld>>>()\n",
              init_grid_size,
              INIT_KERNEL_THREADS,
              (long long)stream);
      #endif

      // Invoke init_kernel to initialize tile descriptors
      THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
        init_grid_size,
        INIT_KERNEL_THREADS,
        0,
        stream)
        .doit(init_kernel,
              tile_state,
              d_keys_in,
              d_keys_prev_in,
              tile_size,
              num_tiles);

      // Check for failure to launch
      error = CubDebug(cudaPeekAtLastError());
      if (cudaSuccess != error)
      {
        break;
      }

      // Sync the stream if specified to flush runtime errors

      error = CubDebug(detail::DebugSyncStream(stream));
      if (cudaSuccess != error)
      {
        break;
      }

      // Get SM occupancy for scan_kernel
      int scan_sm_occupancy;
      error = CubDebug(MaxSmOccupancy(scan_sm_occupancy, // out
                                      scan_kernel,
                                      Policy::BLOCK_THREADS));
      if (cudaSuccess != error)
      {
        break;
      }

      // Get max x-dimension of grid
      int max_dim_x;
      error = CubDebug(cudaDeviceGetAttribute(&max_dim_x, cudaDevAttrMaxGridDimX, device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Run grids in epochs (in case number of tiles exceeds max x-dimension
      int scan_grid_size = CUB_MIN(num_tiles, max_dim_x);
      for (int start_tile = 0; start_tile < num_tiles;
           start_tile += scan_grid_size)
      {
        // Log scan_kernel configuration
        #ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
        _CubLog("Invoking %d DispatchRollingReduce scan_kernel<<<%d, %d, 0, %lld>>>(), %d items "
                "per thread, %d SM occupancy\n",
                start_tile,
                scan_grid_size,
                Policy::BLOCK_THREADS,
                (long long)stream,
                Policy::ITEMS_PER_THREAD,
                scan_sm_occupancy);
        #endif

        // Invoke scan_kernel
        THRUST_NS_QUALIFIER::cuda_cub::launcher::triple_chevron(
          scan_grid_size,
          Policy::BLOCK_THREADS,
          0,
          stream)
          .doit(scan_kernel,
                d_keys_in,
                d_keys_prev_in,
                d_in,
                d_zipped_in,
                d_out,
                tile_state,
                start_tile,
                reduction_op,
                num_items);

        // Check for failure to launch
        error = CubDebug(cudaPeekAtLastError());
        if (cudaSuccess != error)
        {
          break;
        }

        // Sync the stream if specified to flush runtime errors
        error = CubDebug(detail::DebugSyncStream(stream));
        if (cudaSuccess != error)
        {
          break;
        }
      }
    } while (0);

    return error;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION __host__ CUB_FORCE_INLINE cudaError_t Invoke()
  {
    using MaxPolicyT = typename DispatchRollingReduce::MaxPolicy;
    using RollingReduceTileStateT = ReduceByKeyScanTileState<AccumT, OffsetT>;

    // Ensure kernels are instantiated.
    return Invoke<ActivePolicyT>(
      DeviceScanByKeyInitKernel<RollingReduceTileStateT, KeysInputIteratorT>,
      DeviceRollingReduceKernel<MaxPolicyT,
                                KeysInputIteratorT,
                                UnzippedInputIteratorT,
                                ZippedInputIteratorT,
                                OutputIteratorT,
                                RollingReduceTileStateT,
                                UnzippedReductionOpT,
                                OffsetT,
                                UnzippedAccumT>);
  }

  CUB_RUNTIME_FUNCTION CUB_FORCE_INLINE static cudaError_t
  Dispatch(void *d_temp_storage,
           size_t &temp_storage_bytes,
           UnzippedInputIteratorT d_in,
           OutputIteratorT d_out,
           UnzippedReductionOpT reduction_op,
           OffsetT num_items,
           OffsetT window_size,
           cudaStream_t stream)
  {
    using MaxPolicyT = typename DispatchRollingReduce::MaxPolicy;

    cudaError_t error;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      error = CubDebug(PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      // Create dispatch functor
      DispatchRollingReduce dispatch(d_temp_storage,
                                     temp_storage_bytes,
                                     d_in,
                                     d_out,
                                     reduction_op,
                                     num_items,
                                     window_size,
                                     stream,
                                     ptx_version);

      // Dispatch to chained policy
      error = CubDebug(MaxPolicyT::Invoke(ptx_version, dispatch));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);

    return error;
  }

  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED
  CUB_RUNTIME_FUNCTION CUB_FORCE_INLINE static cudaError_t
  Dispatch(void *d_temp_storage,
           size_t &temp_storage_bytes,
           UnzippedInputIteratorT d_in,
           OutputIteratorT d_out,
           UnzippedReductionOpT reduction_op,
           OffsetT num_items,
           OffsetT window_size,
           cudaStream_t stream,
           bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Dispatch(d_temp_storage,
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
