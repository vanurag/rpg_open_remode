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

#include <ros/ros.h>
#include <rmd/check_cuda_device.cuh>
#include <rmd/depthmap_node.h>

#include <vikit/params_helper.h>

int main(int argc, char **argv)
{
  if(!rmd::checkCudaDevice(argc, argv))
    return EXIT_FAILURE;

  ros::init(argc, argv, "rpg_open_remode");
  ros::NodeHandle nh;
  rmd::DepthmapNode dm_node(nh);
  if(!dm_node.init())
  {
    ROS_ERROR("could not initialize DepthmapNode. Shutting down node...");
    return EXIT_FAILURE;
  }

  std::string dense_input_topic;
  if(vk::hasParam("remode/dense_input")) {
    dense_input_topic = vk::getParam<std::string>("remode/dense_input");
  } else {
    dense_input_topic = "/svo/dense_input";
  }
  std::string external_depth_topic;
  bool external_depth_source_available = false;
  if(vk::hasParam("remode/external_depthmap_source")) {
    external_depth_source_available = true;
    external_depth_topic = std::string(vk::getParam<std::string>("remode/external_depthmap_source"));
  }

  ros::Subscriber dense_input_sub;
  typedef message_filters::sync_policies::ApproximateTime<svo_msgs::DenseInput, sensor_msgs::Image> MySyncPolicy;
  message_filters::Subscriber<svo_msgs::DenseInput> mf_sub_dense_input(nh, dense_input_topic, 100);
  message_filters::Subscriber<sensor_msgs::Image> mf_sub_depth(nh, external_depth_topic, 100);
  message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), mf_sub_dense_input, mf_sub_depth);
  if (external_depth_source_available) {
    sync.registerCallback(boost::bind(&rmd::DepthmapNode::denseInputAndExternalDepthCallback, &dm_node, _1, _2));
  } else {
    dense_input_sub = nh.subscribe(
        dense_input_topic,
        1,
        &rmd::DepthmapNode::denseInputCallback,
        &dm_node);
  }

  ros::Rate loop_rate(30);
  while(ros::ok())
  {
    ros::spinOnce();
    loop_rate.sleep();
  }

  return EXIT_SUCCESS;
}
