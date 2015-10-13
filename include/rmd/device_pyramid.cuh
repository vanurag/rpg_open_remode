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

#ifndef RMD_DEVICE_PYRAMID_CUH
#define RMD_DEVICE_PYRAMID_CUH

#include <rmd/device_image.cuh>

namespace rmd
{

void pyrDown(
    const DeviceImage<float> &in_img,
    DeviceImage<float> &out_img);

struct DevicePyramid
{
  __host__
  DevicePyramid(size_t width, size_t height, int num_levels);

  __host__
  ~DevicePyramid();

  __host__
  void setDevData(const float * aligned_data_row_major);

  __host__
  void getDevData(float * aligned_data_row_major, int level) const;

  DeviceImage<float> **images;
  const int num_levels;
};

} // rmd namespace

#endif // RMD_DEVICE_PYRAMID_CUH
