#include "dark_cuda.h"
#include "yolo_v2_class.hpp"

#include <stdio.h>

__global__ void mat_to_image_gpu(float *dst, unsigned char *mat, int w, int h, int step)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	/* convert RGB to BGR, and convert cv::Mat to image_t */
	if (idx < w || idy < h) {
		dst[0 * w * h + idy * w + idx] = mat[idy * step + 3 * idx + 2] / 255.0f;
		dst[1 * w * h + idy * w + idx] = mat[idy * step + 3 * idx + 1] / 255.0f;
		dst[2 * w * h + idy * w + idx] = mat[idy * step + 3 * idx + 0] / 255.0f;
	}
}

int cuda_mat_to_image(image_t *dst, unsigned char *mat, int width, int height, int step)
{
	unsigned char *d_mat;
	float *d_dst;
	unsigned int mat_size = height * step * sizeof(unsigned char);
	unsigned int dst_size = 3 * width * height * sizeof(float);
	CHECK_CUDA(cudaMalloc((unsigned char**)&d_mat, mat_size));
	CHECK_CUDA(cudaMalloc((float**)&d_dst, dst_size));

	float *buf = (float *)malloc(dst_size);

	dim3 dimGrid(ceil(width/(float)32), ceil(height/(float)32));
	dim3 dimBlock(32, 32);

	CHECK_CUDA(cudaMemcpy(d_mat, mat, mat_size, cudaMemcpyHostToDevice));
	mat_to_image_gpu<<<dimGrid, dimBlock>>>(d_dst, d_mat, width, height, step);
	CHECK_CUDA(cudaGetLastError());
	cudaDeviceSynchronize();

	CHECK_CUDA(cudaMemcpy(dst->data, d_dst, dst_size, cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaFree(d_mat));
	CHECK_CUDA(cudaFree(d_dst));

	return 0;
}
