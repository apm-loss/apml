#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/pair.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <vector>
#include <device_launch_parameters.h>


__device__ float compute_distance(const float* a, const float* b, int dim) {
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return sqrtf(dist);
}

__global__ void count_sparse_softmax_nnz(
    int B, int N, int M, int D,
    const float* x,
    const float* y,
    float p_min,
    int* nnz_per_point,
    float threshold = 1e-8f
) {
    int b = blockIdx.x;
    int n = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || n >= N) return;

    const float* x_point = x + b * N * D + n * D;

    float d_min = 1e10f, d_second = 1e10f;
    for (int j = 0; j < M; ++j) {
        const float* y_point = y + b * M * D + j * D;
        float dist = compute_distance(x_point, y_point, D);
        if (dist < d_min) {
            d_second = d_min;
            d_min = dist;
        } else if (dist < d_second) {
            d_second = dist;
        }
    }

    float gap = d_second - d_min;
    float k = (float)M;
    float log_term = -logf((1.0f - p_min) / ((k - 1.0f)));
    float temperature = log_term / fmaxf(gap, 1e-8f);

    int local_count = 0;
    for (int j = 0; j < M; ++j) {
        const float* y_point = y + b * M * D + j * D;
        float dist = compute_distance(x_point, y_point, D);
        float rel_dist = dist - d_min;
        float sim = expf(-rel_dist * temperature);
        if (sim > threshold) local_count++;
    }

    nnz_per_point[b * N + n] = local_count;
}

__global__ void write_sparse_softmax_entries(
    int B, int N, int M, int D,
    const float* x,
    const float* y,
    float p_min,
    const int* nnz_prefix,
    int* sparse_indices,
    int* sparse_row_ids,
    float* sparse_values,
    float threshold = 1e-8f
) {
    int b = blockIdx.x;
    int n = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || n >= N) return;

    const float* x_point = x + b * N * D + n * D;

    float d_min = 1e10f, d_second = 1e10f;
    for (int j = 0; j < M; ++j) {
        const float* y_point = y + b * M * D + j * D;
        float dist = compute_distance(x_point, y_point, D);
        if (dist < d_min) {
            d_second = d_min;
            d_min = dist;
        } else if (dist < d_second) {
            d_second = dist;
        }
    }

    float gap = d_second - d_min;
    float k = (float)M;
    float log_term = -logf((1.0f - p_min) / ((k - 1.0f)));
    float temperature = log_term / fmaxf(gap, 1e-8f);

    int base = nnz_prefix[b * N + n];
    int write_index = 0;
    float denom = 0.0f;
    for (int j = 0; j < M; ++j) {
        const float* y_point = y + b * M * D + j * D;
        float dist = compute_distance(x_point, y_point, D);
        float rel_dist = dist - d_min;
        float sim = expf(-rel_dist * temperature);
        if (sim > threshold) {
            sparse_indices[base + write_index] = b * M + j;
            sparse_row_ids[base + write_index] = b * N + n;
            sparse_values[base + write_index] = sim;
            denom += sim;
            write_index++;
        }
    }
    for (int i = 0; i < write_index; ++i) {
        sparse_values[base + i] /= denom;
    }
}

__global__ void normalize_sparse_kernel(
    int nnz,
    const int* index_ids,   // either COO_i or COO_j
    float* values,          // in-place normalization
    float* norm_buffer      // size = n_rows or n_cols
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;

    int idx = index_ids[tid];
    atomicAdd(&norm_buffer[idx], values[tid]);
}

__global__ void apply_normalization_kernel(
    int nnz,
    const int* index_ids,
    float* values,
    const float* norm_buffer
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;

    float norm = norm_buffer[index_ids[tid]];
    if (norm > 1e-8f) {
        values[tid] /= norm;
    }
}

void sinkhorn_cuda(
    int nnz,
    int n_rows,
    int n_cols,
    int* COO_i,
    int* COO_j,
    float* COO_values,
    int iterations = 20,
    int threads = 256
) {
    float* buffer;
    cudaMalloc(&buffer, std::max(n_rows, n_cols) * sizeof(float));

    int blocks = (nnz + threads - 1) / threads;

    for (int iter = 0; iter < iterations; ++iter) {
        // Row normalize
        cudaMemset(buffer, 0, n_rows * sizeof(float));
        normalize_sparse_kernel<<<blocks, threads>>>(nnz, COO_i, COO_values, buffer);
        apply_normalization_kernel<<<blocks, threads>>>(nnz, COO_i, COO_values, buffer);

        // Column normalize
        cudaMemset(buffer, 0, n_cols * sizeof(float));
        normalize_sparse_kernel<<<blocks, threads>>>(nnz, COO_j, COO_values, buffer);
        apply_normalization_kernel<<<blocks, threads>>>(nnz, COO_j, COO_values, buffer);
    }

    cudaFree(buffer);
}

__global__ void compute_weighted_cost_kernel(
    int B, int N, int M, int D,
    const float* x_ptr, const float* y_ptr,
    const int* COO_i, const int* COO_j,
    const float* COO_values,
    float* loss_buffer,
    int nnz
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;

    int i = COO_i[tid];   // Global i = b * N + i_idx
    int j = COO_j[tid];   // Global j = b * M + j_idx
    int b = i / N;
    int i_idx = i % N;
    int j_idx = j % M;

    const float* x_point = x_ptr + (b * N + i_idx) * D;
    const float* y_point = y_ptr + (b * M + j_idx) * D;

    float dist = 0.0f;
    for (int d = 0; d < D; ++d) {
        float diff = x_point[d] - y_point[d];
        dist += diff * diff;
    }

    loss_buffer[tid] = COO_values[tid] * sqrtf(dist);
}

std::vector<at::Tensor> compute_sparse_softmax_combined(
    const at::Tensor& x,
    const at::Tensor& y,
    float p_min,
    float threshold = 1e-8f
) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int threads = static_cast<int>(0.5f * prop.maxThreadsPerBlock);
    
    int B = x.size(0), N = x.size(1), M = y.size(1), D = x.size(2);
    dim3 blockDim(threads);
    dim3 gridRow(B, (N + threads - 1) / threads);
    dim3 gridCol(B, (M + threads - 1) / threads);

    // === First pass: COUNT non-zeros (row-wise) ===
    at::Tensor row_nnz_per_point = at::zeros({B * N}, x.options().dtype(at::kInt));
    count_sparse_softmax_nnz<<<gridRow, blockDim>>>(
        B, N, M, D,
        x.data_ptr<float>(), y.data_ptr<float>(),
        p_min,
        row_nnz_per_point.data_ptr<int>(),
        threshold
    );

    // === Compute prefix sum ===
    at::Tensor row_prefix = at::zeros_like(row_nnz_per_point);
    thrust::device_ptr<int> row_counts_ptr(row_nnz_per_point.data_ptr<int>());
    thrust::device_ptr<int> row_prefix_ptr(row_prefix.data_ptr<int>());
    thrust::exclusive_scan(row_counts_ptr, row_counts_ptr + B * N, row_prefix_ptr);
    int row_nnz = row_nnz_per_point.sum().item<int>();

    if (row_nnz == 0) {
        at::Tensor empty = at::empty({0}, x.options().dtype(at::kInt));
        at::Tensor zero_loss = at::zeros({1}, x.options().dtype(at::kFloat));
        return {empty.clone(), empty.clone(), at::empty({0}, x.options().dtype(at::kFloat)), zero_loss};
    }

    // === Allocate row-wise sparse tensors ===
    at::Tensor row_sparse_indices = at::empty({row_nnz}, x.options().dtype(at::kInt));
    at::Tensor row_sparse_row_ids = at::empty({row_nnz}, x.options().dtype(at::kInt));
    at::Tensor row_sparse_values = at::empty({row_nnz}, x.options().dtype(at::kFloat));

    // === Second pass: WRITE row-wise ===
    write_sparse_softmax_entries<<<gridRow, blockDim>>>(
        B, N, M, D,
        x.data_ptr<float>(), y.data_ptr<float>(),
        p_min,
        row_prefix.data_ptr<int>(),
        row_sparse_indices.data_ptr<int>(),
        row_sparse_row_ids.data_ptr<int>(),
        row_sparse_values.data_ptr<float>(),
        threshold
    );

    // === Repeat for column-wise ===
    at::Tensor col_nnz_per_point = at::zeros({B * M}, y.options().dtype(at::kInt));
    count_sparse_softmax_nnz<<<gridCol, blockDim>>>(
        B, M, N, D,
        y.data_ptr<float>(), x.data_ptr<float>(),
        p_min,
        col_nnz_per_point.data_ptr<int>(),
        threshold
    );

    at::Tensor col_prefix = at::zeros_like(col_nnz_per_point);
    thrust::device_ptr<int> col_counts_ptr(col_nnz_per_point.data_ptr<int>());
    thrust::device_ptr<int> col_prefix_ptr(col_prefix.data_ptr<int>());
    thrust::exclusive_scan(col_counts_ptr, col_counts_ptr + B * M, col_prefix_ptr);
    int col_nnz = col_nnz_per_point.sum().item<int>();

    at::Tensor col_sparse_indices = at::empty({col_nnz}, x.options().dtype(at::kInt));
    at::Tensor col_sparse_row_ids = at::empty({col_nnz}, x.options().dtype(at::kInt));
    at::Tensor col_sparse_values = at::empty({col_nnz}, x.options().dtype(at::kFloat));

    write_sparse_softmax_entries<<<gridCol, blockDim>>>(
        B, M, N, D,
        y.data_ptr<float>(), x.data_ptr<float>(),
        p_min,
        col_prefix.data_ptr<int>(),
        col_sparse_indices.data_ptr<int>(),
        col_sparse_row_ids.data_ptr<int>(),
        col_sparse_values.data_ptr<float>(),
        threshold
    );

    // === Combine the results ===
    at::Tensor all_i = at::cat({row_sparse_row_ids, col_sparse_indices}, 0);
    at::Tensor all_j = at::cat({row_sparse_indices, col_sparse_row_ids}, 0);
    at::Tensor all_v = at::cat({row_sparse_values, col_sparse_values}, 0);

    // === Merge duplicates (i,j) and average values ===
    // Use thrust (keys = i<<32 | j)
    int total_nnz = all_i.size(0);
    thrust::device_vector<long long> keys(total_nnz);
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(all_i.data_ptr<int>(), all_j.data_ptr<int>())),
        thrust::make_zip_iterator(thrust::make_tuple(all_i.data_ptr<int>(), all_j.data_ptr<int>())) + total_nnz,
        keys.begin(),
        [] __device__ (thrust::tuple<int, int> t) {
            int i = thrust::get<0>(t);
            int j = thrust::get<1>(t);
            return (static_cast<long long>(i) << 32) | static_cast<unsigned int>(j);
        }
    );

    thrust::device_vector<float> values(all_v.data_ptr<float>(), all_v.data_ptr<float>() + total_nnz);

    thrust::device_vector<long long> sorted_keys = keys;
    thrust::device_vector<float> sorted_values = values;
    thrust::sort_by_key(sorted_keys.begin(), sorted_keys.end(), sorted_values.begin());

    thrust::device_vector<long long> unique_keys(total_nnz);
    thrust::device_vector<float> summed_values(total_nnz);
    auto reduce_end = thrust::reduce_by_key(
        sorted_keys.begin(), sorted_keys.end(),
        sorted_values.begin(),
        unique_keys.begin(), summed_values.begin()
    );
    int num_unique = thrust::distance(unique_keys.begin(), reduce_end.first);

    thrust::device_vector<int> counts(total_nnz, 1);
    thrust::reduce_by_key(
        sorted_keys.begin(), sorted_keys.end(),
        counts.begin(),
        unique_keys.begin(), counts.begin()
    );

    thrust::transform(
        summed_values.begin(), summed_values.begin() + num_unique,
        counts.begin(),
        summed_values.begin(),
        [] __device__ (float total, int count) {
            return total / max(1.0f, float(count));
        }
    );

    at::Tensor COO_i = at::empty({num_unique}, x.options().dtype(at::kInt));
    at::Tensor COO_j = at::empty({num_unique}, x.options().dtype(at::kInt));
    at::Tensor COO_v = at::empty({num_unique}, x.options().dtype(at::kFloat));

    // Decode keys
    std::vector<long long> h_keys(num_unique);
    cudaMemcpy(h_keys.data(), thrust::raw_pointer_cast(unique_keys.data()), num_unique * sizeof(long long), cudaMemcpyDeviceToHost);
    std::vector<int> h_i(num_unique), h_j(num_unique);
    for (int i = 0; i < num_unique; ++i) {
        h_i[i] = static_cast<int>(h_keys[i] >> 32);
        h_j[i] = static_cast<int>(h_keys[i] & 0xFFFFFFFF);
    }
    cudaMemcpy(COO_i.data_ptr<int>(), h_i.data(), num_unique * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(COO_j.data_ptr<int>(), h_j.data(), num_unique * sizeof(int), cudaMemcpyHostToDevice);
    thrust::copy(summed_values.begin(), summed_values.begin() + num_unique, COO_v.data_ptr<float>());

    // === Sinkhorn normalization ===
    // sinkhorn_cuda(
    //     num_unique, N, M,
    //     COO_i.data_ptr<int>(), COO_j.data_ptr<int>(), COO_v.data_ptr<float>(),
    //     20, threads
    // );

    sinkhorn_cuda(
        num_unique, B * N, B * M, 
        COO_i.data_ptr<int>(), COO_j.data_ptr<int>(), COO_v.data_ptr<float>(), 
        20, threads
    );


    // === Compute weighted cost ===
    at::Tensor loss_buffer = at::zeros({num_unique}, x.options().dtype(at::kFloat));
    int threads_used = std::min(threads, 512);
    int blocks = (num_unique + threads_used - 1) / threads_used;

    compute_weighted_cost_kernel<<<blocks, threads_used>>>(
        B, N, M, D,
        x.data_ptr<float>(), y.data_ptr<float>(),
        COO_i.data_ptr<int>(), COO_j.data_ptr<int>(), COO_v.data_ptr<float>(),
        loss_buffer.data_ptr<float>(),
        num_unique
    );

    at::Tensor loss = loss_buffer.sum().unsqueeze(0);  // shape [1]
    return {COO_i, COO_j, COO_v, loss};
}
