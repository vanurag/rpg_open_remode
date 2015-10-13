// This file is part of REMODE - REgularized MOnocular Depth Estimation.
//
// Copyright (C) 2014 Matia Pizzoli <matia dot pizzoli at gmail dot com>
// Robotics and Perception Group, University of Zurich, Switzerland
// http://rpg.ifi.uzh.ch
//
// REMODE is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// REMODE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <rmd/device_image.cuh>
#include <rmd/texture_memory.cuh>

namespace rmd
{

__constant__
float c_kernel[5];

extern "C" void setConvolutionKernel(float *h_Kernel)
{
  cudaMemcpyToSymbol(c_kernel, h_Kernel, 5*sizeof(float));
}

__global__
void convolutionRowsKernel(DeviceImage<float> *out_dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  DeviceImage<float> &out_img = *out_dev_ptr;

  if(x >= out_img.width || y >= out_img.height)
    return;

  const float xx = x+0.5f;
  const float yy = y+0.5f;

  float sum = 0.0f;
  for (int k = -2; k <= 2; ++k)
  {
    sum += tex2D(curr_img_tex, xx+static_cast<float>(k), yy) * c_kernel[2-k];
  }
  out_img(x, y) = sum;
  __syncthreads();
}

__global__
void convolutionColsKernel(DeviceImage<float> *out_dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  DeviceImage<float> &out_img = *out_dev_ptr;

  if(x >= out_img.width || y >= out_img.height)
    return;

  const float xx = x+0.5f;
  const float yy = y+0.5f;

  float sum = 0.0f;
  for (int k = -2; k <= 2; ++k)
  {
    sum += tex2D(curr_img_tex, xx, yy+static_cast<float>(k)) * c_kernel[2-k];
  }
  out_img(x, y) = sum;
  __syncthreads();
}

__global__
void halfSampleKernel(DeviceImage<float> *out_dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  DeviceImage<float> &out_img = *out_dev_ptr;

  if(x >= out_img.width || y >= out_img.height)
    return;

  const float xx = x+0.5f;
  const float yy = y+0.5f;

  out_img(x, y) = tex2D(curr_img_tex, xx*2.0f, yy*2.0f);
  __syncthreads();
}

void pyrDown(
    const DeviceImage<float> &in_img,
    DeviceImage<float> &out_img)
{
  // CUDA fields
  dim3 dim_block;
  dim3 dim_grid;
  dim_block.x = 16;
  dim_block.y = 16;
  dim_grid.x = (in_img.width  + dim_block.x - 1) / dim_block.x;
  dim_grid.y = (in_img.height + dim_block.y - 1) / dim_block.y;

  float h_kernel[5] = {1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f};
  setConvolutionKernel(h_kernel);

  rmd::bindTexture(curr_img_tex, in_img, cudaFilterModePoint);
  convolutionRowsKernel<<<dim_grid, dim_block>>>(in_img.dev_ptr);

  rmd::bindTexture(curr_img_tex, in_img, cudaFilterModePoint);
  convolutionColsKernel<<<dim_grid, dim_block>>>(in_img.dev_ptr);

  dim_grid.x = (out_img.width  + dim_block.x - 1) / dim_block.x;
  dim_grid.y = (out_img.height + dim_block.y - 1) / dim_block.y;
  rmd::bindTexture(curr_img_tex, in_img, cudaFilterModePoint);
  halfSampleKernel<<<dim_grid, dim_block>>>(out_img.dev_ptr);

  cudaDeviceSynchronize();
}

} // rmd namespace
