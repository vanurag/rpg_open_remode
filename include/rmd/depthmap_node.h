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

#ifndef RMD_DEPTHMAP_NODE_H
#define RMD_DEPTHMAP_NODE_H

#include <rmd/depthmap.h>

#include <rmd/publisher.h>

#include <svo_msgs/DenseInput.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <ros/ros.h>

#include <opencv2/viz/vizcore.hpp>
#include <opencv2/viz/types.hpp>

namespace rmd
{

namespace ProcessingStates
{
enum State
{
  UPDATE,
  TAKE_REFERENCE_FRAME,
};
}
typedef ProcessingStates::State State;

class DepthmapNode
{
public:
  DepthmapNode(ros::NodeHandle &nh);
  bool init();
  void denseInputCallback(
      const svo_msgs::DenseInputConstPtr &dense_input);
  void denseInputAndExternalDepthCallback(const svo_msgs::DenseInputConstPtr& dense_msg,
                                          const sensor_msgs::ImageConstPtr& depth_msg);
private:
  void denoiseAndPublishResults();
  void publishConvergenceMap();

  // pose viz
  cv::viz::KeyboardEvent viz_key_event;
  static cv::viz::Viz3d viz_window_;
  static cv::Affine3f viz_pose_;
  static void VizKeyboardCallback(const cv::viz::KeyboardEvent&, void*) {
    std::cout << "Setting VIZ viewing angle to camera's viewing direction" << std::endl;
    cv::Affine3f viz_viewer_pose = viz_pose_;
    viz_viewer_pose = viz_viewer_pose.translate(cv::Vec3f(0.0, 0.0, -10.0));
    viz_window_.setViewerPose(viz_viewer_pose);
  }

  float cam_fx_, cam_fy_, cam_cx_, cam_cy_;
  size_t cam_width_, cam_height_;
  std::shared_ptr<rmd::Depthmap> depthmap_;
  State state_;
  float ref_compl_perc_;
  float max_dist_from_ref_;
  int publish_conv_every_n_;
  int num_msgs_;

  ros::NodeHandle &nh_;
  std::unique_ptr<rmd::Publisher> publisher_;

  // external depthmap related
  void transformExternalDepthmap();
  float ext_fx_, ext_fy_, ext_cx_, ext_cy_;
  rmd::SE3<float> ext_cam_to_cam_;
//  image_transport::ImageTransport external_depth_it_;
//  image_transport::Subscriber external_depth_sub_;
//  sensor_msgs::ImageConstPtr external_depth_msg_;
  cv::Mat external_depth_uchar_, external_depth_float_, transformed_external_depth_float_, transformed_external_depth_mask_;
  bool external_depth_available_;
};

} // rmd namespace

#endif // RMD_DEPTHMAP_NODE_H
