#include <span>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <ranges>
#include <iostream>

#include <thrust/universal_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <thrust/allocate_unique.h>

#include "device_rolling_reduce.cuh"

#include <cuda/std/cstddef>

#include <nvbench/nvbench.cuh>
#include <nvbench/main.cuh>

auto rolling_maximum_ranges_cpp20(thrust::universal_vector<int> input, std::size_t window_size) {
  auto const output_size = input.size() - window_size + 1;
  thrust::universal_vector<int> out_vector;
  out_vector.reserve(output_size);
  auto out_it = std::back_insert_iterator(out_vector);
  for (auto idx : std::views::iota(0ULL, output_size)) {
    out_it++ = *std::ranges::max_element(input.begin() + idx, input.begin() + idx + window_size);
  }
  return out_vector;
}

struct local_maximum {
  __host__ __device__
  auto operator()(int idx) const {
    return *thrust::max_element(thrust::seq, input + idx, input + idx + window_size);
  }
  int* const input;
  std::size_t const window_size;
};

struct rolling_maximum_thrust_max_element_t {
  auto operator()(thrust::universal_vector<int> input, std::size_t window_size) const {
    auto const output_size = input.size() - window_size + 1;
    thrust::universal_vector<int> out_vector(output_size);

    thrust::counting_iterator<int> iota_begin(0);
    thrust::counting_iterator<int> iota_end(output_size);

    thrust::transform(iota_begin, iota_end, out_vector.begin(),
      local_maximum{thrust::raw_pointer_cast(input.data()), window_size});

    return out_vector;
  }
};
constexpr rolling_maximum_thrust_max_element_t rolling_maximum_thrust_max_element{};

/*
input      = [a, b, c, d, e, f, g, h, i]
windowed   = [a, b, c] [d, e, f] [g, h, i]

PM = [a, M(a,b), M(a,b,c)] [d, M(d,e), M(d,e,f)] [g, M(g,h), M(g,h,i)]
SM = [M(c,b,a), M(c,b), c] [M(f,e,d), M(f,e), f] [M(i,h,g), M(i,h), i]

max = M(PM[i + k - 1], SM[i])
    = [M(PM[2], SM[0]),       M(PM[3], SM[1]), M(PM[4], SM[2]), ..., M(PM[8], SM[6])]
    = [M(M(a,b,c), M(c,b,a)), M(d, M(c,b)),    M(M(d,e), c),    ..., M(M(g,h,i), M(i,h,g))]
*/

struct index_to_window {
  __host__ __device__ int operator()(int i) const { return i / window_size; }
  std::size_t const window_size;
};

struct rolling_maximum_thrust_scan_t {
  auto operator()(thrust::universal_vector<int> input, std::size_t window_size) const {
    thrust::counting_iterator<int> iota(0);
    thrust::transform_iterator window(iota, index_to_window{window_size});

    thrust::universal_vector<int> prefix_max(input.size());
    thrust::inclusive_scan_by_key(window, window + input.size(), input.begin(), prefix_max.begin(),
                                  thrust::equal_to<int>{}, thrust::maximum<int>{});

    thrust::universal_vector<int> suffix_max(input.size());
    thrust::inclusive_scan_by_key(window, window + input.size(), input.rbegin(), suffix_max.rbegin(),
                                  thrust::equal_to<int>{}, thrust::maximum<int>{});

    thrust::transform(prefix_max.begin() + window_size - 1, prefix_max.end(),
                      suffix_max.begin(), suffix_max.begin(),
                      thrust::maximum<int>{});
    suffix_max.resize(input.size() - window_size + 1);

    return suffix_max;
  }
};
constexpr rolling_maximum_thrust_scan_t rolling_maximum_thrust_scan{};

struct pincer_maximum {
  __host__ __device__
  auto operator()(thrust::tuple<int, int> left, thrust::tuple<int, int> right) const {
    constexpr thrust::maximum<int> mx{};
    return thrust::make_tuple(mx(thrust::get<0>(left), thrust::get<0>(right)),
                              mx(thrust::get<1>(left), thrust::get<1>(right)));
  }
};

struct rolling_maximum_thrust_single_scan_t {
  auto operator()(thrust::universal_vector<int> input, std::size_t window_size) const {
    thrust::counting_iterator<int> const iota(0);
    thrust::transform_iterator const window(iota, index_to_window{window_size});

    auto const pincer_input = thrust::make_zip_iterator(thrust::make_tuple(input.begin(), input.rbegin()));

    thrust::universal_vector<int> prefix_max(input.size());
    thrust::universal_vector<int> suffix_max(input.size());
    auto const pincer_output = thrust::make_zip_iterator(thrust::make_tuple(prefix_max.begin(), suffix_max.rbegin()));

    thrust::inclusive_scan_by_key(window, window + input.size(), pincer_input, pincer_output,
                                  thrust::equal_to<int>{}, pincer_maximum{});

    thrust::transform(prefix_max.begin() + window_size - 1, prefix_max.end(),
                      suffix_max.begin(), suffix_max.begin(),
                      thrust::maximum<int>{});
    suffix_max.resize(input.size() - window_size + 1);

    return suffix_max;
  }
};
constexpr rolling_maximum_thrust_single_scan_t rolling_maximum_thrust_single_scan{};

/*
input      = [a, b, c, d, e, f, g, h, i]
windowed   = [a, b, c] [d, e, f] [g, h, i]

PM =                   [a] [d, M(d,e), M(d,e,f)] [g, M(g,h), M(g,h,i)]
SM = [M(c,b,a), M(c,b), c] [M(f,e,d), M(f,e), f] [g]

max = M(PM[i], SM[i])
    = [M(PM[0], SM[0]), M(PM[1], SM[1]), M(PM[2], SM[2]), ..., M(PM[6], SM[6])]
    = [M(a, M(c,b,a)),  M(d, M(c,b)),    M(M(d,e), c),    ..., M(M(g,h,i), g)]
*/

struct rolling_maximum_thrust_scan_local_t {
  auto operator()(thrust::universal_vector<int> input, std::size_t window_size) const {
    thrust::counting_iterator<int> iota(0);
    thrust::transform_iterator window(iota, index_to_window{window_size});

    thrust::universal_vector<int> prefix_max(input.size() - window_size + 1);
    thrust::inclusive_scan_by_key(window + window_size - 1, window + input.size(),
                                  input.begin() + window_size - 1,
                                  prefix_max.begin(),
                                  thrust::equal_to<int>{}, thrust::maximum<int>{});

    thrust::universal_vector<int> suffix_max(input.size() - window_size + 1);
    thrust::inclusive_scan_by_key(window + window_size - 1, window + input.size(),
                                  input.rbegin() + window_size - 1,
                                  suffix_max.rbegin(),
                                  thrust::equal_to<int>{}, thrust::maximum<int>{});

    thrust::transform(prefix_max.begin(), prefix_max.end(),
                      suffix_max.begin(), prefix_max.begin(),
                      thrust::maximum<int>{});

    return prefix_max;
  }
};
constexpr rolling_maximum_thrust_scan_local_t rolling_maximum_thrust_scan_local{};

struct rolling_maximum_thrust_single_scan_local_t {
  auto operator()(thrust::universal_vector<int> input, std::size_t window_size) const {
    auto const iota = thrust::make_counting_iterator(0);
    auto const window = thrust::make_transform_iterator(iota, index_to_window{window_size});

    auto const pincer_input = thrust::make_zip_iterator(
      thrust::make_tuple(input.begin() + window_size - 1, input.rbegin() + window_size - 1));

    thrust::universal_vector<int> prefix_max(input.size() - window_size + 1);
    thrust::universal_vector<int> suffix_max(input.size() - window_size + 1);
    auto const pincer_output = thrust::make_zip_iterator(thrust::make_tuple(prefix_max.begin(), suffix_max.rbegin()));

    thrust::inclusive_scan_by_key(window + window_size - 1, window + input.size(), pincer_input, pincer_output,
                                  thrust::equal_to<int>{}, pincer_maximum{});

    thrust::transform(prefix_max.begin(), prefix_max.end(),
                      suffix_max.begin(), suffix_max.begin(),
                      thrust::maximum<int>{});

    return suffix_max;
  }
};
constexpr rolling_maximum_thrust_single_scan_local_t rolling_maximum_thrust_single_scan_local{};

struct rolling_maximum_thrust_single_pass_t {
  auto operator()(thrust::universal_vector<int> input, std::size_t window_size) const {
    std::size_t tmp_storage = 0;
    cub::DeviceRollingReduce::RollingReduce(
      NULL, tmp_storage, input.begin(), input.begin(),
      thrust::maximum<int>{}, input.size(), window_size);

    auto tmp = thrust::uninitialized_allocate_unique_n<cuda::std::byte>(
      thrust::universal_allocator<cuda::std::byte>{}, tmp_storage);

    thrust::universal_vector<int> output(input.size() - window_size + 1);
    cub::DeviceRollingReduce::RollingReduce(
      thrust::raw_pointer_cast(tmp.get()), tmp_storage, input.begin(), output.begin(),
      thrust::maximum<int>{}, input.size(), window_size);

    cudaError_t const error = cudaDeviceSynchronize();
    if (error != cudaSuccess) throw false;

    return output;
  }
};
constexpr rolling_maximum_thrust_single_pass_t rolling_maximum_thrust_single_pass{};

using algorithms = nvbench::type_list<
/*
  rolling_maximum_thrust_max_element_t,
  rolling_maximum_thrust_scan_t,
  rolling_maximum_thrust_single_scan_t,
  rolling_maximum_thrust_scan_local_t,
  rolling_maximum_thrust_single_scan_local_t,
*/
  rolling_maximum_thrust_single_pass_t
>;

template <typename F>
void debug_rolling_maximum(F f, thrust::universal_vector<int>& input, std::size_t window_size) {
  for (auto&& i: f(input, window_size))
    std::cout << i << " ";
  std::cout << "\n";
}

template <typename F>
void test_rolling_maximum(F f, thrust::universal_vector<int>& input, std::size_t window_size) {
  auto gold = rolling_maximum_ranges_cpp20(input, window_size);
  auto output = f(input, window_size);
  if (gold.size() != output.size() || !thrust::equal(gold.begin(), gold.end(), output.begin())) {
    std::cout << "Test failed, input size: " << input.size()
              << ", window size: " << window_size << "\n";

    if (input.size() < 1024) {
      std::cout << "input:  ";
      for (auto&& i: input) std::cout << i << " ";
      std::cout << "\n";

      std::cout << "gold:   ";
      for (auto&& i: gold) std::cout << i << " ";
      std::cout << "\n";

      std::cout << "output: ";
      for (auto&& i: output) std::cout << i << " ";
      std::cout << "\n";
    } else {
      for (auto idx : std::views::iota(0ULL, gold.size())) {
        if (gold[idx] != output[idx])
          std::cout << "Mismatch at index " << idx
                    << ", gold: " << gold[idx]
                    << ", output: " << output[idx] << "\n";
      }
    }

    throw false;
  }
}

template <typename Algorithm, typename... Tail>
void test_all_rolling_maximum(nvbench::type_list<Algorithm, Tail...>,
                              thrust::universal_vector<int>& input, std::size_t window_size) {
  test_rolling_maximum(Algorithm{}, input, window_size);
  if constexpr (sizeof...(Tail) > 0)
    test_all_rolling_maximum(nvbench::type_list<Tail...>{}, input, window_size);
}

void test_all_rolling_maximum(thrust::universal_vector<int> input, std::size_t window_size) {
  test_all_rolling_maximum(algorithms{}, input, window_size);
}

auto generate_input(std::size_t problem_size) {
  std::mt19937 rng(1337);
  std::uniform_int_distribution<> dist(0, 100);

  thrust::universal_vector<int> input(problem_size);
  std::generate(input.begin(), input.end(), [&] { return dist(rng); });

  return std::move(input);
}

auto generate_iota(std::size_t problem_size, int init) {
  thrust::universal_vector<int> input(problem_size);
  thrust::sequence(input.rbegin(), input.rend(), init);
  return std::move(input);
}

template <typename F>
void benchmark_rolling_maximum(nvbench::state& state, nvbench::type_list<F>) {
  std::size_t const problem_size = state.get_int64("ProblemSize");
  std::size_t const window_size  = state.get_int64("WindowSize");

  auto input = generate_input(problem_size);

  cudaStream_t default_stream = 0;
  state.set_cuda_stream(nvbench::make_cuda_stream_view(default_stream));

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    F{}(input, window_size);
  });
}

NVBENCH_BENCH_TYPES(benchmark_rolling_maximum, NVBENCH_TYPE_AXES(algorithms))
  .set_type_axes_names({"Algorithms"})
  .add_int64_axis("ProblemSize", {1024 * 1024 * 16})
  .add_int64_axis("WindowSize", {2, 3, 64, 512, 1024, 1024 * 16})
;

int main(int argc, char const* const* argv) {
  test_all_rolling_maximum({3, 8, 1, 2}, 2);
  test_all_rolling_maximum({1, 6, 3, 8, 9, 6, 5, 4, 3}, 3);
  test_all_rolling_maximum({1, 2, 3, 6, 5, 4, 7, 8, 9}, 3);
  test_all_rolling_maximum({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 2);
  test_all_rolling_maximum(generate_input(1024), 2);
  test_all_rolling_maximum(generate_input(1024), 4);
  test_all_rolling_maximum(generate_input(1024), 64);
  test_all_rolling_maximum(generate_iota(4096, 1), 2);
  test_all_rolling_maximum(generate_input(16384 * 16), 2);

  //NVBENCH_MAIN_BODY(argc, argv);
}
