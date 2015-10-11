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

#include <gtest/gtest.h>
#include <cuda_toolkit/helper_timer.h>
#include <opencv2/opencv.hpp>

#include "device_pyramid_test.cuh"

TEST(RMDCuTests, downSampleTest)
{
  cv::Mat img = cv::imread("../test_data/images/scene_000.png", CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img_flt;
  img.convertTo(img_flt, CV_32F, 1./255.);

  const size_t orig_w = img_flt.cols;
  const size_t orig_h = img_flt.rows;

  // Opencv downsample
  cv::Mat ocv_down_sampled;
  double t = (double)cv::getTickCount();
  cv::pyrDown(img_flt, ocv_down_sampled);
  t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
  printf("Opencv execution time: %f seconds.\n", t);

  const size_t new_ocv_w = ocv_down_sampled.cols;
  const size_t new_ocv_h = ocv_down_sampled.rows;
  printf("Size of original image (width, height):\n(%lu, %lu)\nSize of OpenCV downsampled image:\n(%lu, %lu)\n",
         orig_w, orig_h, new_ocv_w, new_ocv_h);

  // CUDA downsample

  // upload data to device memory
  rmd::DeviceImage<float> in_img(orig_w, orig_h);
  in_img.setDevData(reinterpret_cast<float*>(img_flt.data));

  // prepare output image
  rmd::DeviceImage<float> out_img((orig_w+1)/2, (orig_h+1)/2);

  StopWatchInterface * timer = NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);
  sdkStartTimer(&timer);
  rmd::pyrDown(in_img, out_img);
  sdkStopTimer(&timer);
  t = sdkGetAverageTimerValue(&timer) / 1000.0;
  printf("CUDA execution time: %f seconds.\n", t);

  // Download result to host
  cv::Mat cu_down_sampled(out_img.height, out_img.width, CV_32FC1);
  out_img.getDevData(reinterpret_cast<float*>(cu_down_sampled.data));

  cv::imshow("CUDA Downsampled", cu_down_sampled);
  cv::waitKey();
}
