/******************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
 * @file AgentRollingReduce implements a stateful abstraction of CUDA thread blocks
 *       for participating in device-wide rolling reductions.
 */

#pragma once

#define CUB_FORCE_INLINE __forceinline__

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/config.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_type.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <iterator>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/

// TODO: Am I actually using this? We piggyback on the ScanByKey tunings, so I'm
// not sure if this is needed.
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

/******************************************************************************
 * Function objects
 ******************************************************************************/

template <typename OffsetT>
struct IndexToWindow
{
  __host__ __device__ OffsetT
  operator()(OffsetT i) const
  {
    return i / window_size;
  }

  // TODO: Pass this as a compile time parameter?
  OffsetT const window_size;
};

template <typename ReductionOpT>
struct ZippedReduce {
  template <typename T>
  __host__ __device__ thrust::tuple<T, T>
  operator()(thrust::tuple<T, T> const& left, thrust::tuple<T, T> const& right) const
  {
    if (threadIdx.x == 0)
    printf("prefix: %d %d, suffix %d %d, result %d %d\n", thrust::get<0>(left), thrust::get<0>(right), thrust::get<1>(left), thrust::get<1>(right),
           op(thrust::get<0>(left), thrust::get<0>(right)),
           op(thrust::get<1>(left), thrust::get<1>(right))
           );
    return thrust::make_tuple(op(thrust::get<0>(left), thrust::get<0>(right)),
                              op(thrust::get<1>(left), thrust::get<1>(right)));
  }

  ReductionOpT op;
};

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

/**
 * @brief AgentRollingReduce implements a stateful abstraction of CUDA thread
 *        blocks for participating in device-wide rolling reductions.
 *
 * @tparam AgentRollingReducePolicyT
 *   Parameterized AgentScanPolicyT tuning policy type
 *
 * @tparam UnzippedInputIteratorT
 *   Random-access input iterator type
 *
 * @tparam OutputIteratorT
 *   Random-access output iterator type
 *
 * @tparam UnzippedReductionOpT
 *   Scan functor type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam UnzippedAccumT
 *   The type of intermediate accumulator (according to P2322R6)
 */
template <typename AgentRollingReducePolicyT,
          typename UnzippedInputIteratorT,
          typename OutputIteratorT,
          typename UnzippedReductionOpT,
          typename OffsetT,
          typename UnzippedAccumT>
struct AgentRollingReduce
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  using KeysInputIteratorT   = thrust::transform_iterator<
    IndexToWindow<OffsetT>, thrust::counting_iterator<OffsetT>>;
  using ZippedInputIteratorT = thrust::zip_iterator<
    thrust::tuple<UnzippedInputIteratorT, thrust::reverse_iterator<UnzippedInputIteratorT>>>;

  using AccumT = thrust::tuple<UnzippedAccumT, UnzippedAccumT>;

  using KeyT               = OffsetT;
  using InputT             = cub::detail::value_t<ZippedInputIteratorT>;
  using UnzippedInputT     = cub::detail::value_t<UnzippedInputIteratorT>;
  using SizeValuePairT     = KeyValuePair<OffsetT, AccumT>;

  using EqualityOpT        = cub::Equality;
  using ReductionOpT       = ZippedReduce<UnzippedReductionOpT>;
  using ReduceBySegmentOpT = ReduceBySegmentOp<ReductionOpT>;

  using ScanTileStateT = ReduceByKeyScanTileState<AccumT, OffsetT>;

  // Constants
  static constexpr int BLOCK_THREADS  = AgentRollingReducePolicyT::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD =
    AgentRollingReducePolicyT::ITEMS_PER_THREAD;
  static constexpr int ITEMS_PER_TILE = BLOCK_THREADS * ITEMS_PER_THREAD;

  using WrappedKeysInputIteratorT = cub::detail::conditional_t<
    std::is_pointer<KeysInputIteratorT>::value,
    CacheModifiedInputIterator<AgentRollingReducePolicyT::LOAD_MODIFIER, KeyT, OffsetT>,
    KeysInputIteratorT>;

  using WrappedZippedInputIteratorT = cub::detail::conditional_t<
    std::is_pointer<ZippedInputIteratorT>::value,
    CacheModifiedInputIterator<AgentRollingReducePolicyT::LOAD_MODIFIER,
                               InputT,
                               OffsetT>,
    ZippedInputIteratorT>;

  using BlockLoadKeysT = BlockLoad<KeyT,
                                   BLOCK_THREADS,
                                   ITEMS_PER_THREAD,
                                   AgentRollingReducePolicyT::LOAD_ALGORITHM>;

  using BlockLoadValuesT = BlockLoad<AccumT,
                                     BLOCK_THREADS,
                                     ITEMS_PER_THREAD,
                                     AgentRollingReducePolicyT::LOAD_ALGORITHM>;

  using BlockStoreValuesT = BlockStore<UnzippedAccumT,
                                       BLOCK_THREADS,
                                       ITEMS_PER_THREAD,
                                       AgentRollingReducePolicyT::STORE_ALGORITHM>;

  using BlockDiscontinuityKeysT = BlockDiscontinuity<KeyT, BLOCK_THREADS, 1, 1>;

  using DelayConstructorT = typename AgentRollingReducePolicyT::detail::delay_constructor_t;
  using TilePrefixCallbackT =
    TilePrefixCallbackOp<SizeValuePairT, ReduceBySegmentOpT, ScanTileStateT, 0, DelayConstructorT>;

  using BlockScanT = BlockScan<SizeValuePairT,
                               BLOCK_THREADS,
                               AgentRollingReducePolicyT::SCAN_ALGORITHM,
                               1,
                               1>;

  union TempStorage_
  {
    struct ScanStorage
    {
      typename BlockScanT::TempStorage scan;
      typename TilePrefixCallbackT::TempStorage prefix;
      typename BlockDiscontinuityKeysT::TempStorage discontinuity;
    } scan_storage;

    typename BlockLoadKeysT::TempStorage load_keys;
    typename BlockLoadValuesT::TempStorage load_values;
    typename BlockStoreValuesT::TempStorage store_values;
  };

  struct TempStorage : cub::Uninitialized<TempStorage_>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  TempStorage_ &storage;
  WrappedKeysInputIteratorT d_keys_in;
  KeyT *d_keys_prev_in;
  UnzippedInputIteratorT d_in;
  WrappedZippedInputIteratorT d_zipped_in;
  OutputIteratorT d_out;
  InequalityWrapper<EqualityOpT> inequality_op;
  UnzippedReductionOpT unzipped_reduce_op;
  ReductionOpT reduce_op;
  ReduceBySegmentOpT pair_reduce_op;

  //---------------------------------------------------------------------
  // Block scan utility methods (first tile)
  //---------------------------------------------------------------------

  __device__ CUB_FORCE_INLINE void
  ScanTile(SizeValuePairT (&scan_items)[ITEMS_PER_THREAD],
           SizeValuePairT &tile_aggregate)
  {
    BlockScanT(storage.scan_storage.scan)
      .InclusiveScan(scan_items, scan_items, pair_reduce_op, tile_aggregate);
  }

  //---------------------------------------------------------------------
  // Block scan utility methods (subsequent tiles)
  //---------------------------------------------------------------------

  __device__ CUB_FORCE_INLINE void
  ScanTile(SizeValuePairT (&scan_items)[ITEMS_PER_THREAD],
           SizeValuePairT &tile_aggregate,
           TilePrefixCallbackT &prefix_op)
  {
    BlockScanT(storage.scan_storage.scan)
      .InclusiveScan(scan_items, scan_items, pair_reduce_op, prefix_op);
    tile_aggregate = prefix_op.GetBlockAggregate();
  }

  //---------------------------------------------------------------------
  // Zip utility methods
  //---------------------------------------------------------------------

  template <bool IS_LAST_TILE>
  __device__ CUB_FORCE_INLINE void
  ZipValuesAndFlags(OffsetT num_remaining,
                    AccumT  (&values)[ITEMS_PER_THREAD],
                    OffsetT (&segment_flags)[ITEMS_PER_THREAD],
                    SizeValuePairT (&scan_items)[ITEMS_PER_THREAD])
  {
    // Zip values and segment_flags
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      // Set segment_flags for first out-of-bounds item, zero for others
      if (IS_LAST_TILE &&
          OffsetT(threadIdx.x * ITEMS_PER_THREAD) + ITEM == num_remaining)
      {
        segment_flags[ITEM] = 1;
      }

      scan_items[ITEM].value = values[ITEM];
      scan_items[ITEM].key   = segment_flags[ITEM];
    }
  }

  __device__ CUB_FORCE_INLINE void
  UnzipValues(AccumT         (&values)[ITEMS_PER_THREAD],
              SizeValuePairT (&scan_items)[ITEMS_PER_THREAD])
  {
    // Unzip values and segment_flags
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      values[ITEM] = scan_items[ITEM].value;
    }
  }

  __device__ CUB_FORCE_INLINE void
  FinalReduce(AccumT         (&values)[ITEMS_PER_THREAD],
              UnzippedAccumT (&final)[ITEMS_PER_THREAD])
  {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      if (threadIdx.x == 0)
      printf("item: %d, left: %d, right %d\n", ITEM, thrust::get<0>(values[ITEM]), thrust::get<1>(values[ITEM]));
      final[ITEM] = unzipped_reduce_op(
        thrust::get<0>(values[ITEM]), thrust::get<1>(values[ITEM]));
    }
  }

  //---------------------------------------------------------------------
  // Cooperatively scan a device-wide sequence of tiles with other CTAs
  //---------------------------------------------------------------------

  // Process a tile of input (dynamic chained scan)
  //
  template <bool IS_LAST_TILE>
  __device__ CUB_FORCE_INLINE void ConsumeTile(OffsetT /*num_items*/,
                                              OffsetT num_remaining,
                                              int tile_idx,
                                              OffsetT tile_base,
                                              ScanTileStateT &tile_state)
  {
    // Load items
    KeyT   keys[ITEMS_PER_THREAD];
    AccumT values[ITEMS_PER_THREAD];

    if (IS_LAST_TILE)
    {
      // Fill last element with the first element
      // because collectives are not suffix guarded
      BlockLoadKeysT(storage.load_keys)
        .Load(d_keys_in + tile_base,
              keys,
              num_remaining,
              -1/**(d_keys_in + tile_base)*/);
    }
    else
    {
      BlockLoadKeysT(storage.load_keys).Load(d_keys_in + tile_base, keys);
    }

    CTA_SYNC();

    if (IS_LAST_TILE)
    {
      // Fill last element with the first element
      // because collectives are not suffix guarded
      BlockLoadValuesT(storage.load_values)
        .Load(d_zipped_in + tile_base,
              values,
              num_remaining,
              -1/**(d_zipped_in + tile_base)*/);
    }
    else
    {
      BlockLoadValuesT(storage.load_values)
        .Load(d_zipped_in + tile_base, values);
    }

    CTA_SYNC();

    OffsetT segment_flags[ITEMS_PER_THREAD];
    SizeValuePairT scan_items[ITEMS_PER_THREAD];

    // first tile
    if (tile_idx == 0)
    {
      BlockDiscontinuityKeysT(storage.scan_storage.discontinuity)
        .FlagHeads(segment_flags, keys, inequality_op);

      // Zip values and segment_flags
      ZipValuesAndFlags<IS_LAST_TILE>(num_remaining,
                                      values,
                                      segment_flags,
                                      scan_items);

      // Exclusive scan of values and segment_flags
      SizeValuePairT tile_aggregate;
      ScanTile(scan_items, tile_aggregate);

      if (threadIdx.x == 0)
      {
        if (!IS_LAST_TILE)
        {
          tile_state.SetInclusive(0, tile_aggregate);
        }

        scan_items[0].key = 0;
      }
    }
    else
    {
      KeyT tile_pred_key = (threadIdx.x == 0) ? d_keys_prev_in[tile_idx]
                                              : KeyT();

      BlockDiscontinuityKeysT(storage.scan_storage.discontinuity)
        .FlagHeads(segment_flags, keys, inequality_op, tile_pred_key);

      // Zip values and segment_flags
      ZipValuesAndFlags<IS_LAST_TILE>(num_remaining,
                                      values,
                                      segment_flags,
                                      scan_items);

      SizeValuePairT tile_aggregate;
      TilePrefixCallbackT prefix_op(tile_state,
                                    storage.scan_storage.prefix,
                                    pair_reduce_op,
                                    tile_idx);
      ScanTile(scan_items, tile_aggregate, prefix_op);
    }

    CTA_SYNC();

    UnzipValues(values, scan_items);

    UnzippedAccumT final[ITEMS_PER_THREAD];
    FinalReduce(values, final);

    // Store items
    if (IS_LAST_TILE)
    {
      BlockStoreValuesT(storage.store_values)
        .Store(d_out + tile_base, final, num_remaining);
    }
    else
    {
      BlockStoreValuesT(storage.store_values)
        .Store(d_out + tile_base, final);
    }
  }

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  // Dequeue and scan tiles of items as part of a dynamic chained scan
  // with Init functor
  __device__ CUB_FORCE_INLINE AgentRollingReduce(TempStorage &storage,
                                                KeysInputIteratorT d_keys_in,
                                                KeyT *d_keys_prev_in,
                                                UnzippedInputIteratorT d_in,
                                                ZippedInputIteratorT d_zipped_in,
                                                OutputIteratorT d_out,
                                                UnzippedReductionOpT reduce_op)
      : storage(storage.Alias())
      , d_keys_in(d_keys_in)
      , d_keys_prev_in(d_keys_prev_in)
      , d_in(d_in)
      , d_zipped_in(d_zipped_in)
      , d_out(d_out)
      , inequality_op{EqualityOpT{}}
      , unzipped_reduce_op{reduce_op}
      , reduce_op{ReductionOpT{reduce_op}}
      , pair_reduce_op{ReductionOpT{reduce_op}}
  {}

  /**
   * Scan tiles of items as part of a dynamic chained scan
   *
   * @param num_items
   *   Total number of input items
   *
   * @param tile_state
   *   Global tile state descriptor
   *
   * start_tile
   *   The starting tile for the current grid
   */
  __device__ CUB_FORCE_INLINE void ConsumeRange(OffsetT num_items,
                                               ScanTileStateT &tile_state,
                                               int start_tile)
  {
    int tile_idx          = blockIdx.x;
    OffsetT tile_base     = OffsetT(ITEMS_PER_TILE) * tile_idx;
    OffsetT num_remaining = num_items - tile_base;

    if (num_remaining > ITEMS_PER_TILE)
    {
      // Not the last tile (full)
      ConsumeTile<false>(num_items,
                         num_remaining,
                         tile_idx,
                         tile_base,
                         tile_state);
    }
    else if (num_remaining > 0)
    {
      // The last tile (possibly partially-full)
      ConsumeTile<true>(num_items,
                        num_remaining,
                        tile_idx,
                        tile_base,
                        tile_state);
    }
  }
};

CUB_NAMESPACE_END

