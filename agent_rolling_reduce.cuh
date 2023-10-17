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

// TODO: Move this into a utility header.
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

#include <cuda/std/atomic>

#include <iterator>

CUB_NAMESPACE_BEGIN

/******************************************************************************
 * Store-release/load-acquire scan-by-key tile status type
 ******************************************************************************/

template <typename KeyT,
          typename ValueT,
          cuda::memory_order MemoryOrder = cuda::memory_order_acq_rel,
          bool SINGLE_WORD = (Traits<ValueT>::PRIMITIVE) && (sizeof(ValueT) + sizeof(KeyT) < 8)>
// TODO: Support 128-bit atomics.
struct SynchronizingScanByKeyTileState;

/**
 * Tile status interface for reduction by key, specialized for scan status and value types that
 * cannot be combined into one machine word.
 */
template <typename KeyT,
          typename ValueT,
          cuda::memory_order MemoryOrder>
struct SynchronizingScanByKeyTileState<KeyT, ValueT, MemoryOrder, false>
{
    using T = KeyValuePair<KeyT, ValueT>;

    // Status word type
    using StatusWord = unsigned int;

    using AtomicRefStatusWord = cuda::atomic_ref<StatusWord, cuda::thread_scope_device>;

    static constexpr cuda::memory_order LoadMemoryOrder =
      MemoryOrder == cuda::memory_order_acq_rel ?
      cuda::memory_order_acquire :
      MemoryOrder;

    static constexpr cuda::memory_order StoreMemoryOrder =
      MemoryOrder == cuda::memory_order_acq_rel ?
      cuda::memory_order_release :
      MemoryOrder;

    // Constants
    enum
    {
        TILE_STATUS_PADDING = CUB_PTX_WARP_THREADS,
    };

    // Device storage
    StatusWord *d_tile_status;
    T          *d_tile_partial;
    T          *d_tile_inclusive;

    /// Constructor
    __host__ __device__ CUB_FORCE_INLINE
    SynchronizingScanByKeyTileState()
    :
        d_tile_status(nullptr),
        d_tile_partial(nullptr),
        d_tile_inclusive(nullptr)
    {}


    /// Initializer
    __host__ __device__ CUB_FORCE_INLINE
    cudaError_t Init(
        int     num_tiles,                          ///< [in] Number of tiles
        void    *d_temp_storage,                    ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t  temp_storage_bytes)                 ///< [in] Size in bytes of \t d_temp_storage allocation
    {
        cudaError_t error = cudaSuccess;
        do
        {
            void*   allocations[3] = {};
            size_t  allocation_sizes[3];

            allocation_sizes[0] = (num_tiles + TILE_STATUS_PADDING) * sizeof(StatusWord);           // bytes needed for tile status descriptors
            allocation_sizes[1] = (num_tiles + TILE_STATUS_PADDING) * sizeof(Uninitialized<T>);     // bytes needed for partials
            allocation_sizes[2] = (num_tiles + TILE_STATUS_PADDING) * sizeof(Uninitialized<T>);     // bytes needed for inclusives

            // Compute allocation pointers into the single storage blob
            error = CubDebug(
              AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));

            if (cudaSuccess != error)
            {
              break;
            }

            // Alias the offsets
            d_tile_status       = reinterpret_cast<StatusWord*>(allocations[0]);
            d_tile_partial      = reinterpret_cast<T*>(allocations[1]);
            d_tile_inclusive    = reinterpret_cast<T*>(allocations[2]);
        }
        while (0);

        return error;
    }


    /**
     * Compute device memory needed for tile status
     */
    __host__ __device__ CUB_FORCE_INLINE
    static cudaError_t AllocationSize(
        int     num_tiles,                          ///< [in] Number of tiles
        size_t  &temp_storage_bytes)                ///< [out] Size in bytes of \t d_temp_storage allocation
    {
        // Specify storage allocation requirements
        size_t  allocation_sizes[3];
        allocation_sizes[0] = (num_tiles + TILE_STATUS_PADDING) * sizeof(StatusWord);         // bytes needed for tile status descriptors
        allocation_sizes[1] = (num_tiles + TILE_STATUS_PADDING) * sizeof(Uninitialized<T>);   // bytes needed for partials
        allocation_sizes[2] = (num_tiles + TILE_STATUS_PADDING) * sizeof(Uninitialized<T>);   // bytes needed for inclusives

        // Set the necessary size of the blob
        void* allocations[3] = {};
        return CubDebug(AliasTemporaries(NULL, temp_storage_bytes, allocations, allocation_sizes));
    }


    /**
     * Initialize (from device)
     */
    __device__ CUB_FORCE_INLINE void InitializeStatus(int num_tiles)
    {
        int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
        if (tile_idx < num_tiles)
        {
            // Not-yet-set
            d_tile_status[TILE_STATUS_PADDING + tile_idx] = StatusWord(SCAN_TILE_INVALID);
        }

        if ((blockIdx.x == 0) && (threadIdx.x < TILE_STATUS_PADDING))
        {
            // Padding
            d_tile_status[threadIdx.x] = StatusWord(SCAN_TILE_OOB);
        }
    }


    /**
     * Update the specified tile's inclusive value and corresponding status
     */
    __device__ CUB_FORCE_INLINE void SetInclusive(int tile_idx, T tile_inclusive)
    {
        ThreadStore<STORE_CG>(d_tile_inclusive + TILE_STATUS_PADDING + tile_idx, tile_inclusive);
        AtomicRefStatusWord alias(d_tile_status[TILE_STATUS_PADDING + tile_idx]);
        alias.store(StatusWord(SCAN_TILE_INCLUSIVE), StoreMemoryOrder);
    }


    /**
     * Update the specified tile's partial value and corresponding status
     */
    __device__ CUB_FORCE_INLINE void SetPartial(int tile_idx, T tile_partial)
    {
        // Update tile partial value
        ThreadStore<STORE_CG>(d_tile_partial + TILE_STATUS_PADDING + tile_idx, tile_partial);
        AtomicRefStatusWord alias(d_tile_status[TILE_STATUS_PADDING + tile_idx]);
        alias.store(StatusWord(SCAN_TILE_PARTIAL), StoreMemoryOrder);
    }

    /**
     * Wait for the corresponding tile to become non-invalid
     */
    template <class DelayT = detail::default_no_delay_t>
    __device__ CUB_FORCE_INLINE void WaitForValid(
        int             tile_idx,
        StatusWord      &status,
        T               &value,
        DelayT          delay = {})
    {
        do
        {
          delay(); // TODO: Use `atomic::wait/notify`.
          AtomicRefStatusWord alias(d_tile_status[TILE_STATUS_PADDING + tile_idx]);
          status = alias.load(LoadMemoryOrder);
        } while (WARP_ANY((status == SCAN_TILE_INVALID), 0xffffffff));

        if (status == StatusWord(SCAN_TILE_PARTIAL))
        {
          value = ThreadLoad<LOAD_CG>(d_tile_partial + TILE_STATUS_PADDING + tile_idx);
        }
        else
        {
          value = ThreadLoad<LOAD_CG>(d_tile_inclusive + TILE_STATUS_PADDING + tile_idx);
        }
    }

    /**
     * Loads and returns the tile's value. The returned value is undefined if either (a) the tile's status is invalid or
     * (b) there is no memory fence between reading a non-invalid status and the call to LoadValid.
     */
    __device__ CUB_FORCE_INLINE T LoadValid(int tile_idx)
    {
        return d_tile_inclusive[TILE_STATUS_PADDING + tile_idx];
    }
};


/**
 * Tile status interface for reduction by key, specialized for scan status and value types that
 * can be combined into one machine word that can be read/written coherently in a single access.
 */
template <typename KeyT,
          typename ValueT,
          cuda::memory_order MemoryOrder>
struct SynchronizingScanByKeyTileState<KeyT, ValueT, MemoryOrder, true>
{
    using T = KeyValuePair<KeyT, ValueT>;

    static constexpr cuda::memory_order LoadMemoryOrder =
      MemoryOrder == cuda::memory_order_acq_rel ?
      cuda::memory_order_acquire :
      MemoryOrder;

    static constexpr cuda::memory_order StoreMemoryOrder =
      MemoryOrder == cuda::memory_order_acq_rel ?
      cuda::memory_order_release :
      MemoryOrder;

    // Constants
    enum
    {
        PAIR_SIZE           = static_cast<int>(sizeof(ValueT) + sizeof(KeyT)),
        TXN_WORD_SIZE       = 1 << Log2<PAIR_SIZE + 1>::VALUE,
        STATUS_WORD_SIZE    = TXN_WORD_SIZE - PAIR_SIZE,

        TILE_STATUS_PADDING = CUB_PTX_WARP_THREADS,
    };

    // Status word type
    using StatusWord = cub::detail::conditional_t<
      STATUS_WORD_SIZE == 8,
      unsigned long long,
      cub::detail::conditional_t<
        STATUS_WORD_SIZE == 4,
        unsigned int,
        cub::detail::conditional_t<STATUS_WORD_SIZE == 2, unsigned short, unsigned char>>>;

    // Status word type
    using TxnWord = cub::detail::conditional_t<
      TXN_WORD_SIZE == 16,
      ulonglong2,
      cub::detail::conditional_t<TXN_WORD_SIZE == 8, unsigned long long, unsigned int>>;

    // Device word type (for when sizeof(ValueT) == sizeof(KeyT))
    struct TileDescriptorBigStatus
    {
        KeyT        key;
        ValueT      value;
        StatusWord  status;
    };

    // Device word type (for when sizeof(ValueT) != sizeof(KeyT))
    struct TileDescriptorLittleStatus
    {
        ValueT      value;
        StatusWord  status;
        KeyT        key;
    };

    // Device word type
    using TileDescriptor =
      cub::detail::conditional_t<sizeof(ValueT) == sizeof(KeyT),
                                 TileDescriptorBigStatus,
                                 TileDescriptorLittleStatus>;

    using AtomicRefTileDescriptor = cuda::atomic_ref<TileDescriptor, cuda::thread_scope_device>;

    // Device storage
    TileDescriptor *d_tile_descriptors;

    /// Constructor
    __host__ __device__ CUB_FORCE_INLINE
    SynchronizingScanByKeyTileState()
    :
        d_tile_descriptors(nullptr)
    {}


    /// Initializer
    __host__ __device__ CUB_FORCE_INLINE
    cudaError_t Init(
        int     /*num_tiles*/,                      ///< [in] Number of tiles
        void    *d_temp_storage,                    ///< [in] Device-accessible allocation of temporary storage.  When NULL, the required allocation size is written to \p temp_storage_bytes and no work is done.
        size_t  /*temp_storage_bytes*/)             ///< [in] Size in bytes of \t d_temp_storage allocation
    {
        d_tile_descriptors = reinterpret_cast<TileDescriptor*>(d_temp_storage);
        return cudaSuccess;
    }


    /**
     * Compute device memory needed for tile status
     */
    __host__ __device__ CUB_FORCE_INLINE
    static cudaError_t AllocationSize(
        int     num_tiles,                          ///< [in] Number of tiles
        size_t  &temp_storage_bytes)                ///< [out] Size in bytes of \t d_temp_storage allocation
    {
        temp_storage_bytes = (num_tiles + TILE_STATUS_PADDING) * sizeof(TileDescriptor); // bytes needed for tile status descriptors
        return cudaSuccess;
    }


    /**
     * Initialize (from device)
     */
    __device__ CUB_FORCE_INLINE void InitializeStatus(int num_tiles)
    {
        int tile_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

        if (tile_idx < num_tiles)
        {
            // Not-yet-set
            d_tile_descriptors[TILE_STATUS_PADDING + tile_idx].status = StatusWord(SCAN_TILE_INVALID);
        }

        if ((blockIdx.x == 0) && (threadIdx.x < TILE_STATUS_PADDING))
        {
            // Padding
            d_tile_descriptors[threadIdx.x].status = StatusWord(SCAN_TILE_OOB);
        }
    }


    /**
     * Update the specified tile's inclusive value and corresponding status
     */
    __device__ CUB_FORCE_INLINE void SetInclusive(int tile_idx, T tile_inclusive)
    {
        TileDescriptor tile_descriptor;
        tile_descriptor.status = SCAN_TILE_INCLUSIVE;
        tile_descriptor.value  = tile_inclusive.value;
        tile_descriptor.key    = tile_inclusive.key;

        AtomicRefTileDescriptor alias(d_tile_descriptors[TILE_STATUS_PADDING + tile_idx]);
        alias.store(tile_descriptor, StoreMemoryOrder);
    }


    /**
     * Update the specified tile's partial value and corresponding status
     */
    __device__ CUB_FORCE_INLINE void SetPartial(int tile_idx, T tile_partial)
    {
        TileDescriptor tile_descriptor;
        tile_descriptor.status = SCAN_TILE_PARTIAL;
        tile_descriptor.value  = tile_partial.value;
        tile_descriptor.key    = tile_partial.key;

        AtomicRefTileDescriptor alias(d_tile_descriptors[TILE_STATUS_PADDING + tile_idx]);
        alias.store(tile_descriptor, StoreMemoryOrder);
    }

    /**
     * Wait for the corresponding tile to become non-invalid
     */
    template <class DelayT = detail::fixed_delay_constructor_t<350, 450>::delay_t>
    __device__ CUB_FORCE_INLINE void WaitForValid(
        int         tile_idx,
        StatusWord  &status,
        T           &value,
        DelayT      delay_or_prevent_hoisting = {})
    {
        TileDescriptor tile_descriptor;

        do
        {
          delay_or_prevent_hoisting(); // TODO: Use `atomic::wait/notify`.
          AtomicRefTileDescriptor alias(d_tile_descriptors[TILE_STATUS_PADDING + tile_idx]);
          tile_descriptor = alias.load(LoadMemoryOrder);
        } while (WARP_ANY((tile_descriptor.status == SCAN_TILE_INVALID), 0xffffffff));

        status      = tile_descriptor.status;
        value.value = tile_descriptor.value;
        value.key   = tile_descriptor.key;
    }
};

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
 * @tparam InputIteratorT
 *   Random-access input iterator type
 *
 * @tparam OutputIteratorT
 *   Random-access output iterator type
 *
 * @tparam ReductionOpT
 *   Scan functor type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam AccumT
 *   The type of intermediate accumulator (according to P2322R6)
 */
template <typename AgentRollingReducePolicyT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename ReductionOpT,
          typename OffsetT,
          typename AccumT>
struct AgentRollingReduce
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  using KeysInputIteratorT     = thrust::transform_iterator<
    IndexToWindow<OffsetT>, thrust::counting_iterator<OffsetT>>;
  using ReverseInputIteratorT  = thrust::reverse_iterator<InputIteratorT>;
  using ReverseOutputIteratorT = thrust::reverse_iterator<OutputIteratorT>;

  using KeyT               = OffsetT;
  using InputT             = cub::detail::value_t<InputIteratorT>;
  using SizeValuePairT     = KeyValuePair<OffsetT, AccumT>;

  using EqualityOpT        = cub::Equality;
  using ReduceBySegmentOpT = ReduceBySegmentOp<ReductionOpT>;

  using SuffixScanTileStateT = SynchronizingScanByKeyTileState<AccumT, OffsetT, cuda::memory_order_relaxed>;
  using PrefixScanTileStateT = SynchronizingScanByKeyTileState<AccumT, OffsetT, cuda::memory_order_acq_rel>;

  // Constants
  static constexpr int BLOCK_THREADS  = AgentRollingReducePolicyT::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD =
    AgentRollingReducePolicyT::ITEMS_PER_THREAD;
  static constexpr int ITEMS_PER_TILE = BLOCK_THREADS * ITEMS_PER_THREAD;

  using WrappedKeysInputIteratorT = cub::detail::conditional_t<
    std::is_pointer<KeysInputIteratorT>::value,
    CacheModifiedInputIterator<AgentRollingReducePolicyT::LOAD_MODIFIER, KeyT, OffsetT>,
    KeysInputIteratorT>;

  using WrappedInputIteratorT = cub::detail::conditional_t<
    std::is_pointer<InputIteratorT>::value,
    CacheModifiedInputIterator<AgentRollingReducePolicyT::LOAD_MODIFIER,
                               InputT,
                               OffsetT>,
    InputIteratorT>;

  using WrappedReverseInputIteratorT = cub::detail::conditional_t<
    std::is_pointer<ReverseInputIteratorT>::value,
    CacheModifiedInputIterator<AgentRollingReducePolicyT::LOAD_MODIFIER,
                               InputT,
                               OffsetT>,
    ReverseInputIteratorT>;

  using BlockLoadKeysT = BlockLoad<KeyT,
                                   BLOCK_THREADS,
                                   ITEMS_PER_THREAD,
                                   AgentRollingReducePolicyT::LOAD_ALGORITHM>;

  using BlockLoadValuesT = BlockLoad<AccumT,
                                     BLOCK_THREADS,
                                     ITEMS_PER_THREAD,
                                     AgentRollingReducePolicyT::LOAD_ALGORITHM>;

  using BlockStoreValuesT = BlockStore<AccumT,
                                       BLOCK_THREADS,
                                       ITEMS_PER_THREAD,
                                       AgentRollingReducePolicyT::STORE_ALGORITHM>;

  using BlockDiscontinuityKeysT = BlockDiscontinuity<KeyT, BLOCK_THREADS, 1, 1>;

  using DelayConstructorT = typename AgentRollingReducePolicyT::detail::delay_constructor_t;
  using SuffixTileCallbackT =
    TilePrefixCallbackOp<SizeValuePairT, ReduceBySegmentOpT, SuffixScanTileStateT, 0, DelayConstructorT>;
  using PrefixTileCallbackT =
    TilePrefixCallbackOp<SizeValuePairT, ReduceBySegmentOpT, PrefixScanTileStateT, 0, DelayConstructorT>;

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
      typename SuffixTileCallbackT::TempStorage suffix_lookback;
      typename PrefixTileCallbackT::TempStorage prefix_lookback;
      typename BlockDiscontinuityKeysT::TempStorage discontinuity;
    } scan_storage;

    typename BlockLoadKeysT::TempStorage load_keys;
    typename BlockLoadValuesT::TempStorage load_suffix;
    typename BlockLoadValuesT::TempStorage load_prefix;
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
  WrappedInputIteratorT d_in;
  WrappedReverseInputIteratorT d_reverse_in;
  OutputIteratorT d_out;
  ReverseOutputIteratorT d_reverse_out;
  InequalityWrapper<EqualityOpT> inequality_op;
  ReductionOpT reduce_op;
  ReduceBySegmentOpT pair_reduce_op;

  //---------------------------------------------------------------------
  // Block scan utility methods (first tile)
  //---------------------------------------------------------------------

  // TODO: Refactor more code into here and align the interface.
  template <bool IS_LAST_TILE, typename ScanTileStateT>
  __device__ CUB_FORCE_INLINE void
  ScanFirstTile(SizeValuePairT (&scan_items)[ITEMS_PER_THREAD],
                ScanTileStateT &tile_state)
  {
    SizeValuePairT tile_aggregate;

    BlockScanT(storage.scan_storage.scan)
      .InclusiveScan(scan_items, scan_items, pair_reduce_op, tile_aggregate);

    if (threadIdx.x == 0)
    {
      if (!IS_LAST_TILE)
      {
        tile_state.SetInclusive(0, tile_aggregate);
      }

      scan_items[0].key = 0;
    }
  }

  //---------------------------------------------------------------------
  // Block scan utility methods (subsequent tiles)
  //---------------------------------------------------------------------

  // TODO: Refactor more code into here and align the interface.
  template <bool IS_LAST_TILE, typename LookbackOp>
  __device__ CUB_FORCE_INLINE void
  ScanSubsequentTile(SizeValuePairT (&scan_items)[ITEMS_PER_THREAD],
                     LookbackOp &lookback_op)
  {
    SizeValuePairT tile_aggregate;
    BlockScanT(storage.scan_storage.scan)
      .InclusiveScan(scan_items, scan_items, pair_reduce_op, lookback_op);
    tile_aggregate = lookback_op.GetBlockAggregate();
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
  FinalReduce(AccumT (&suffix)[ITEMS_PER_THREAD],
              AccumT (&prefix)[ITEMS_PER_THREAD],
              AccumT (&final)[ITEMS_PER_THREAD])
  {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM)
    {
      final[ITEM] = reduce_op(suffix[ITEM], prefix[ITEM]);
    }
  }

  //---------------------------------------------------------------------
  // Cooperatively scan a device-wide sequence of tiles with other CTAs
  //---------------------------------------------------------------------

  // Process a tile of input (dynamic chained scan)
  template <bool IS_LAST_TILE>
  __device__ CUB_FORCE_INLINE void ConsumeTile(OffsetT /*num_items*/,
                                               OffsetT num_remaining,
                                               int tile_idx,
                                               OffsetT tile_base,
                                               SuffixScanTileStateT &suffix_tile_state,
                                               PrefixScanTileStateT &prefix_tile_state)
  {
    ///////////////////////////////////////////////////////////////////////////
    // Load items

    KeyT   keys[ITEMS_PER_THREAD];
    AccumT suffix[ITEMS_PER_THREAD];
    AccumT prefix[ITEMS_PER_THREAD];

    if (IS_LAST_TILE)
    {
      // Fill last element with the first element because collectives are not guarded
      BlockLoadKeysT(storage.load_keys)
        .Load(d_keys_in + tile_base,
              keys,
              num_remaining,
              *(d_keys_in + tile_base));
    }
    else
    {
      BlockLoadKeysT(storage.load_keys).Load(d_keys_in + tile_base, keys);
    }

    CTA_SYNC();

    if (IS_LAST_TILE)
    {
      // Fill last element with the first element because collectives are not guarded

      BlockLoadValuesT(storage.load_suffix)
        .Load(d_reverse_in + tile_base,
              suffix,
              num_remaining,
              *(d_reverse_in + tile_base));

      BlockLoadValuesT(storage.load_prefix)
        .Load(d_in + tile_base,
              prefix,
              num_remaining,
              *(d_in + tile_base));
    }
    else
    {
      BlockLoadValuesT(storage.load_suffix)
        .Load(d_reverse_in + tile_base, suffix);

      BlockLoadValuesT(storage.load_prefix)
        .Load(d_in + tile_base, prefix);
    }

    CTA_SYNC();

    ///////////////////////////////////////////////////////////////////////////
    // Suffix scan with relaxed semantics

    OffsetT segment_flags[ITEMS_PER_THREAD];
    SizeValuePairT scan_items[ITEMS_PER_THREAD];

    if (tile_idx == 0) // First tile
    {
      BlockDiscontinuityKeysT(storage.scan_storage.discontinuity)
        .FlagHeads(segment_flags, keys, inequality_op);

      ZipValuesAndFlags<IS_LAST_TILE>(num_remaining,
                                      suffix,
                                      segment_flags,
                                      scan_items);

      ScanFirstTile<IS_LAST_TILE>(scan_items, suffix_tile_state);
    }
    else
    {
      KeyT tile_pred_key = (threadIdx.x == 0) ? d_keys_prev_in[tile_idx]
                                              : KeyT();

      BlockDiscontinuityKeysT(storage.scan_storage.discontinuity)
        .FlagHeads(segment_flags, keys, inequality_op, tile_pred_key);

      ZipValuesAndFlags<IS_LAST_TILE>(num_remaining,
                                      suffix,
                                      segment_flags,
                                      scan_items);

      SuffixTileCallbackT lookback_op(suffix_tile_state,
                                      storage.scan_storage.suffix_lookback,
                                      pair_reduce_op,
                                      tile_idx);
      ScanSubsequentTile<IS_LAST_TILE>(scan_items, lookback_op);
    }

    CTA_SYNC();

    UnzipValues(suffix, scan_items);

    ///////////////////////////////////////////////////////////////////////////
    // Store suffix scan reversed in output

    if (IS_LAST_TILE)
    {
      BlockStoreValuesT(storage.store_values)
        .Store(d_reverse_out + tile_base, suffix, num_remaining);
    }
    else
    {
      BlockStoreValuesT(storage.store_values)
        .Store(d_reverse_out + tile_base, suffix);
    }

    CTA_SYNC();

    ///////////////////////////////////////////////////////////////////////////
    // Suffix scan with acquire/release semantics. This ensures that the store
    // of the suffix scan to the input will be visible after the prefix scan.

    if (tile_idx == 0) // First tile
    {
      ZipValuesAndFlags<IS_LAST_TILE>(num_remaining,
                                      prefix,
                                      segment_flags,
                                      scan_items);

      ScanFirstTile<IS_LAST_TILE>(scan_items, prefix_tile_state);
    }
    else
    {
      ZipValuesAndFlags<IS_LAST_TILE>(num_remaining,
                                      prefix,
                                      segment_flags,
                                      scan_items);

      PrefixTileCallbackT lookback_op(prefix_tile_state,
                                      storage.scan_storage.prefix_lookback,
                                      pair_reduce_op,
                                      tile_idx);
      ScanSubsequentTile<IS_LAST_TILE>(scan_items, lookback_op);
    }

    CTA_SYNC();

    UnzipValues(prefix, scan_items);

    ///////////////////////////////////////////////////////////////////////////
    // Load suffix scan from output

    if (IS_LAST_TILE)
    {
      // Fill last element with the first element because collectives are not guarded
      BlockLoadValuesT(storage.load_suffix)
        .Load(d_out + tile_base,
              suffix,
              num_remaining,
              *(d_out + tile_base));
    }
    else
    {
      BlockLoadValuesT(storage.load_suffix)
        .Load(d_out + tile_base, suffix);
    }

    CTA_SYNC();

    ///////////////////////////////////////////////////////////////////////////
    // Compute the final results using the prefix and suffix scans

    AccumT final[ITEMS_PER_THREAD];
    FinalReduce(prefix, suffix, final);

    ///////////////////////////////////////////////////////////////////////////
    // Store final result

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
                                                 InputIteratorT d_in,
                                                 ReverseInputIteratorT d_reverse_in,
                                                 OutputIteratorT d_out,
                                                 ReverseOutputIteratorT d_reverse_out,
                                                 ReductionOpT reduce_op)
      : storage(storage.Alias())
      , d_keys_in(d_keys_in)
      , d_keys_prev_in(d_keys_prev_in)
      , d_in(d_in)
      , d_reverse_in(d_reverse_in)
      , d_out(d_out)
      , d_reverse_out(d_reverse_out)
      , inequality_op{EqualityOpT{}}
      , reduce_op{reduce_op}
      , pair_reduce_op{reduce_op}
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
                                                SuffixScanTileStateT &suffix_tile_state,
                                                PrefixScanTileStateT &prefix_tile_state,
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
                         suffix_tile_state,
                         prefix_tile_state);
    }
    else if (num_remaining > 0)
    {
      // The last tile (possibly partially-full)
      ConsumeTile<true>(num_items,
                        num_remaining,
                        tile_idx,
                        tile_base,
                        suffix_tile_state,
                        prefix_tile_state);
    }
  }
};

CUB_NAMESPACE_END

