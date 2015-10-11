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
#include "test_texture_memory.cuh"

namespace rmd
{

__global__
void pyrDownKernel(DeviceImage<float> *out_dev_ptr)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  DeviceImage<float> &out_img = *out_dev_ptr;

  if(x >= out_img.width
     || y >= out_img.height
     || x < 0
     || y < 0)
    return;

  const float xx = x+0.5f;
  const float yy = y+0.5f;

  out_img(x, y) = tex2D(img_tex, xx*2.0f, yy*2.0f);
}

void pyrDown(
    const DeviceImage<float> &in_img,
    DeviceImage<float> &out_img)
{
  rmd::bindTexture(img_tex, in_img);

  // CUDA fields
  dim3 dim_block;
  dim3 dim_grid;
  dim_block.x = 16;
  dim_block.y = 16;
  dim_grid.x = (in_img.width  + dim_block.x - 1) / dim_block.x;
  dim_grid.y = (in_img.height + dim_block.y - 1) / dim_block.y;
  pyrDownKernel<<<dim_grid, dim_block>>>(out_img.dev_ptr);
  cudaDeviceSynchronize();
}

} // rmd namespace
