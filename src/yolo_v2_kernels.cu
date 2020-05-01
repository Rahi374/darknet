#include "dark_cuda.h"
#include "pipeline.hpp"
#include "yolo_v2_class.hpp"

#include <stdio.h>

__global__ void mat_to_image_resize_gpu(float *dst, int dst_w, int dst_h, unsigned char *src, int src_w, int src_h, int src_step)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	int sx = (double)(idx + 0.5) * ((double)src_w / (double)dst_w);
	int sy = (double)(idy + 0.5) * ((double)src_h / (double)dst_h);

	/* subsample, convert RGB to BGR, and convert cv::Mat to image_t */
	if (idx < dst_w || idy < dst_h) {
		dst[0 * dst_w * dst_h + idy * dst_w + idx] = src[sy * src_step + 3 * sx + 2] / 255.0f;
		dst[1 * dst_w * dst_h + idy * dst_w + idx] = src[sy * src_step + 3 * sx + 1] / 255.0f;
		dst[2 * dst_w * dst_h + idy * dst_w + idx] = src[sy * src_step + 3 * sx + 0] / 255.0f;
	}
}

struct dist_index_pair {
	double dist;
	int index;
};

__global__ void do_tracking_gpu(/* rw */ bbox_t *cur_bbox_vec, int cur_bbox_len,
				/* ro */ bool const change_history, int const frames_story, int const max_dist,
				/* ro */ bbox_t *prev_bbox_vec, int *prev_bbox_vec_offsets, int *prev_bbox_vec_sizes,
				/* ro */ int prev_bbox_count, int prev_bbox_max_len,
				/* ws */ dist_index_pair *dist_index_pairs /* w = cur_bbox_len, h = prev_bbox_max_len */,
				/* ro */ int prev_bbox_vec_index)
{
	int idx = blockDim.x * threadIdx.y + threadIdx.x;
	int idy = blockIdx.y;

	if (idx < cur_bbox_len && idy < prev_bbox_vec_sizes[prev_bbox_vec_index]) {
		bbox_t my_cur_bbox = cur_bbox_vec[idx];
		bbox_t my_prev_bbox = prev_bbox_vec[prev_bbox_vec_offsets[prev_bbox_vec_index] + idy];

		dist_index_pair *dip = &dist_index_pairs[idy * cur_bbox_len + idx];
		dip->index = idx;

		/* get distance */
		if ((my_cur_bbox.track_id != 0 && my_cur_bbox.track_id != my_prev_bbox.track_id) ||
				my_cur_bbox.obj_id != my_prev_bbox.obj_id) {
			dip->dist = DBL_MAX;
		} else if (my_cur_bbox.track_id == my_prev_bbox.track_id &&
				my_cur_bbox.obj_id == my_prev_bbox.obj_id) {
			/*
			 * this will go to the front of the list after sorting, and will indicate
			 * that this old track id is already taken, so don't match it
			 */
			dip->dist = -1;
		} else {
			double center_x_diff = (double)(my_prev_bbox.x + my_prev_bbox.w / 2) -
				(double)(my_cur_bbox.x + my_cur_bbox.w / 2);
			double center_y_diff = (double)(my_prev_bbox.y + my_prev_bbox.h / 2) -
				(double)(my_cur_bbox.y + my_cur_bbox.h / 2);
			double dist = sqrt(center_x_diff * center_x_diff + center_y_diff * center_y_diff);
			dip->dist = (dist > max_dist) ? DBL_MAX : dist;
		}

		/* sort distances with odd-even transposition sort */
		__syncthreads();
		dist_index_pair *row_begin = &dist_index_pairs[idy * cur_bbox_len];
		for (int i = 0; i < cur_bbox_len; i++) {
			if ( ((i % 2 == 0) && (idx % 2 == 0) && (idx + 1 < cur_bbox_len)) ||
					((i % 2 == 1) && (idx % 2 == 1) && (idx + 1 < cur_bbox_len)) ) {
				if (row_begin[idx].dist > row_begin[idx + 1].dist) {
					dist_index_pair tmp = row_begin[idx];
					row_begin[idx] = row_begin[idx + 1];
					row_begin[idx + 1] = tmp;
				}
			}
			__syncthreads();
		}

		/* cas assign track id to cur bboxes */
		if (idx == 0) {
			for (int i = 0; i < cur_bbox_len; i++) {
				int cur_bbox_index = row_begin[i].index;
				if (row_begin[i].dist < 0)
					break;
				/* invalid cur_bbox index, or old track id is already taken*/
				if (cur_bbox_index < cur_bbox_len && cur_bbox_index >= 0 &&
						row_begin[i].dist <= max_dist) {
					/* first try to cas in the track id */
					unsigned int ret = atomicCAS(&(cur_bbox_vec[cur_bbox_index].track_id),
							0, my_prev_bbox.track_id);
					if (ret == 0) {
						/* then write in the adjusted w and h */
						cur_bbox_vec[cur_bbox_index].w = (cur_bbox_vec[cur_bbox_index].w + my_prev_bbox.w) / 2;
						cur_bbox_vec[cur_bbox_index].h = (cur_bbox_vec[cur_bbox_index].h + my_prev_bbox.h) / 2;
						break;
					}
				}
			}
		}

	}

}


int cuda_mat_to_image_resize(cudaEvent_t *out_event, void **out_ptr, int dst_w, int dst_h,
			     unsigned char *src, int src_w, int src_h, int src_step)
{
	unsigned char *d_src;
	float *d_dst;
	CHECK_CUDA(cudaEventCreate(out_event));

	unsigned int src_size = src_h * src_step * sizeof(unsigned char);
	unsigned int dst_size = 3 * dst_w * dst_h * sizeof(float);

	CHECK_CUDA(cudaMalloc((unsigned char**)&d_src, src_size));
	CHECK_CUDA(cudaMalloc((float**)&d_dst, dst_size));
	*out_ptr = (void *)d_dst;

	dim3 dimGrid(ceil(dst_w/(float)32), ceil(dst_h/(float)32));
	dim3 dimBlock(32, 32);

	CHECK_CUDA(cudaMemcpyAsync(d_src, src, src_size, cudaMemcpyHostToDevice, get_cuda_stream()));
	mat_to_image_resize_gpu<<<dimGrid, dimBlock, 0, get_cuda_stream()>>>(d_dst, dst_w, dst_h, d_src, src_w, src_h, src_step);
	CHECK_CUDA(cudaGetLastError());
	CHECK_CUDA(cudaEventRecord(*out_event, get_cuda_stream()));

	CHECK_CUDA(cudaFree(d_src));

	return 0;
}

int cuda_do_tracking(/* rw */ bbox_t *cur_bbox_vec, int cur_bbox_len,
		     /* ro */ bool const change_history, int const frames_story, int const max_dist,
		     /* ro */ bbox_t **prev_bbox_vec, int *prev_bbox_vec_offsets, int *prev_bbox_vec_sizes,
		     /* ro */ int prev_bbox_count, int prev_bbox_max_len, int prev_bbox_total_len)
{
	bbox_t *cur_bbox_vec_gpu = NULL;
	unsigned int cur_bbox_vec_memsize = cur_bbox_len * sizeof(bbox_t);
	CHECK_CUDA(cudaMalloc((bbox_t**)&cur_bbox_vec_gpu, cur_bbox_vec_memsize));
	CHECK_CUDA(cudaMemcpyAsync(cur_bbox_vec_gpu, cur_bbox_vec, cur_bbox_vec_memsize,
				   cudaMemcpyHostToDevice, get_cuda_stream()));

	bbox_t *prev_bbox_vec_gpu = NULL;
	unsigned int prev_bbox_vec_memsize = prev_bbox_total_len * sizeof(bbox_t);
	CHECK_CUDA(cudaMalloc((bbox_t**)&prev_bbox_vec_gpu, prev_bbox_vec_memsize));
	for (int i = 0; i < prev_bbox_count; i++)
		CHECK_CUDA(cudaMemcpyAsync(prev_bbox_vec_gpu + prev_bbox_vec_offsets[i],
					   prev_bbox_vec[i], prev_bbox_vec_sizes[i] * sizeof(bbox_t),
					   cudaMemcpyHostToDevice, get_cuda_stream()));

	int *prev_bbox_vec_offsets_gpu = NULL;
	unsigned int prev_bbox_vec_offsets_memsize = prev_bbox_count * sizeof(int);
	CHECK_CUDA(cudaMalloc((int**)&prev_bbox_vec_offsets_gpu, prev_bbox_vec_offsets_memsize));
	CHECK_CUDA(cudaMemcpyAsync(prev_bbox_vec_offsets_gpu, prev_bbox_vec_offsets, prev_bbox_vec_offsets_memsize,
				   cudaMemcpyHostToDevice, get_cuda_stream()));

	int *prev_bbox_vec_sizes_gpu = NULL;
	unsigned int prev_bbox_vec_sizes_memsize = prev_bbox_count * sizeof(int);
	CHECK_CUDA(cudaMalloc((int**)&prev_bbox_vec_sizes_gpu, prev_bbox_vec_sizes_memsize));
	CHECK_CUDA(cudaMemcpyAsync(prev_bbox_vec_sizes_gpu, prev_bbox_vec_sizes, prev_bbox_vec_sizes_memsize,
				   cudaMemcpyHostToDevice, get_cuda_stream()));

	/* this is just working space for the kernel */
	dist_index_pair *dist_index_pair_vec_gpu = NULL;
	unsigned int dist_vec_gpu_memsize = cur_bbox_len * prev_bbox_max_len * sizeof(dist_index_pair);
	CHECK_CUDA(cudaMalloc((dist_index_pair **)&dist_index_pair_vec_gpu, dist_vec_gpu_memsize));

	unsigned int num_threads = 16;
	dim3 dimGrid(ceil(cur_bbox_len/(float)(num_threads*num_threads)), prev_bbox_max_len);
	dim3 dimBlock(num_threads, num_threads);

	for (int i = 0; i < prev_bbox_count; i++) {
		do_tracking_gpu<<<dimGrid, dimBlock, 0, get_cuda_stream()>>>
				(/* rw */ cur_bbox_vec_gpu, cur_bbox_len,
				 /* ro */ change_history, frames_story, max_dist,
				 /* ro */ prev_bbox_vec_gpu, prev_bbox_vec_offsets_gpu, prev_bbox_vec_sizes_gpu,
				 /* ro */ prev_bbox_count, prev_bbox_max_len,
				 /* ws */ dist_index_pair_vec_gpu,
				 /* ro */ i);
		CHECK_CUDA(cudaGetLastError());
		cudaStreamSynchronize(get_cuda_stream());
	}

	CHECK_CUDA(cudaMemcpyAsync(cur_bbox_vec, cur_bbox_vec_gpu, cur_bbox_vec_memsize,
				   cudaMemcpyDeviceToHost, get_cuda_stream()));
	cudaStreamSynchronize(get_cuda_stream());

	CHECK_CUDA(cudaFree(cur_bbox_vec_gpu));
	CHECK_CUDA(cudaFree(prev_bbox_vec_gpu));
	CHECK_CUDA(cudaFree(prev_bbox_vec_offsets_gpu));
	CHECK_CUDA(cudaFree(prev_bbox_vec_sizes_gpu));
	CHECK_CUDA(cudaFree(dist_index_pair_vec_gpu));

	return 0;
}
