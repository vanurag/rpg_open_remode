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

#include <rmd/depthmap_node.h>

#include <rmd/se3.cuh>

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <vikit/params_helper.h>

#include <future>

cv::viz::Viz3d rmd::DepthmapNode::viz_window_ = cv::viz::Viz3d("Dense Input Pose");
cv::Affine3f rmd::DepthmapNode::viz_pose_ = cv::Affine3f();

rmd::DepthmapNode::DepthmapNode(ros::NodeHandle &nh)
  : nh_(nh),
    num_msgs_(0),
    viz_key_event(cv::viz::KeyboardEvent::Action::KEY_DOWN, "A", cv::viz::KeyboardEvent::ALT, 1)
{
  state_ = rmd::State::TAKE_REFERENCE_FRAME;

  // external depth source
  external_depth_available_ = false;
}

bool rmd::DepthmapNode::init()
{
  if(!vk::hasParam("remode/cam_width"))
    return false;
  if(!vk::hasParam("remode/cam_height"))
    return false;
  if(!vk::hasParam("remode/cam_fx"))
    return false;
  if(!vk::hasParam("remode/cam_fy"))
    return false;
  if(!vk::hasParam("remode/cam_cx"))
    return false;
  if(!vk::hasParam("remode/cam_cy"))
    return false;

  cam_width_  = vk::getParam<int>("remode/cam_width");
  cam_height_ = vk::getParam<int>("remode/cam_height");
  cam_fx_     = vk::getParam<float>("remode/cam_fx");
  cam_fy_     = vk::getParam<float>("remode/cam_fy");
  cam_cx_     = vk::getParam<float>("remode/cam_cx");
  cam_cy_     = vk::getParam<float>("remode/cam_cy");

  depthmap_ = std::make_shared<rmd::Depthmap>(cam_width_,
                                              cam_height_,
                                              cam_fx_,
                                              cam_cx_,
                                              cam_fy_,
                                              cam_cy_);

  if(vk::hasParam("remode/cam_k1") &&
     vk::hasParam("remode/cam_k2") &&
     vk::hasParam("remode/cam_r1") &&
     vk::hasParam("remode/cam_r2") )
  {
    depthmap_->initUndistortionMap(
          vk::getParam<float>("remode/cam_k1"),
          vk::getParam<float>("remode/cam_k2"),
          vk::getParam<float>("remode/cam_r1"),
          vk::getParam<float>("remode/cam_r2"));
    cv::Mat new_cam_K = depthmap_->getK();
    cam_fx_ = new_cam_K.at<float>(0, 0);
    cam_cx_ = new_cam_K.at<float>(0, 2);
    cam_fy_ = new_cam_K.at<float>(1, 1);
    cam_cy_ = new_cam_K.at<float>(1, 2);
  }

  if (vk::hasParam("remode/external_depthmap_source")) {
    if(!vk::hasParam("remode/external_cam_fx"))
      return false;
    if(!vk::hasParam("remode/external_cam_fy"))
      return false;
    if(!vk::hasParam("remode/external_cam_cx"))
      return false;
    if(!vk::hasParam("remode/external_cam_cy"))
      return false;
    if(!vk::hasParam("remode/external_cam_to_cam_00"))
      return false;
    if(!vk::hasParam("remode/external_cam_to_cam_10"))
      return false;
    if(!vk::hasParam("remode/external_cam_to_cam_20"))
      return false;
    if(!vk::hasParam("remode/external_cam_to_cam_30"))
      return false;
    if(!vk::hasParam("remode/external_cam_to_cam_01"))
      return false;
    if(!vk::hasParam("remode/external_cam_to_cam_11"))
      return false;
    if(!vk::hasParam("remode/external_cam_to_cam_21"))
      return false;
    if(!vk::hasParam("remode/external_cam_to_cam_31"))
      return false;
    if(!vk::hasParam("remode/external_cam_to_cam_02"))
      return false;
    if(!vk::hasParam("remode/external_cam_to_cam_12"))
      return false;
    if(!vk::hasParam("remode/external_cam_to_cam_22"))
      return false;
    if(!vk::hasParam("remode/external_cam_to_cam_32"))
      return false;

    ext_fx_ = vk::getParam<float>("remode/external_cam_fx");
    ext_fy_ = vk::getParam<float>("remode/external_cam_fy");
    ext_cx_ = vk::getParam<float>("remode/external_cam_cx");
    ext_cy_ = vk::getParam<float>("remode/external_cam_cy");
    float R[9];
    float t[3];
    R[0] = vk::getParam<float>("remode/external_cam_to_cam_00");
    R[1] = vk::getParam<float>("remode/external_cam_to_cam_10");
    R[2] = vk::getParam<float>("remode/external_cam_to_cam_20");
    t[0] = vk::getParam<float>("remode/external_cam_to_cam_30");
    R[3] = vk::getParam<float>("remode/external_cam_to_cam_01");
    R[4] = vk::getParam<float>("remode/external_cam_to_cam_11");
    R[5] = vk::getParam<float>("remode/external_cam_to_cam_21");
    t[1] = vk::getParam<float>("remode/external_cam_to_cam_31");
    R[6] = vk::getParam<float>("remode/external_cam_to_cam_02");
    R[7] = vk::getParam<float>("remode/external_cam_to_cam_12");
    R[8] = vk::getParam<float>("remode/external_cam_to_cam_22");
    t[2] = vk::getParam<float>("remode/external_cam_to_cam_32");
    rmd::SE3<float> T(R, t);
    ext_cam_to_cam_ = T;
  }

  ref_compl_perc_    = vk::getParam<float>("remode/ref_compl_perc",   10.0f);
  max_dist_from_ref_ = vk::getParam<float>("remode/max_dist_from_ref", 0.5f);
  publish_conv_every_n_ = vk::getParam<float>("remode/publish_conv_every_n", 10);

  publisher_.reset(new rmd::Publisher(nh_, depthmap_));

  // VIZ
  viz_window_.registerKeyboardCallback(VizKeyboardCallback);
  viz_window_.setWindowSize(cv::Size(600, 600));
  viz_window_.showWidget("Dense Input Pose", cv::viz::WCoordinateSystem(100.0));

  return true;
}

void rmd::DepthmapNode::denseInputCallback(
    const svo_msgs::DenseInputConstPtr &dense_input)
{
  std::cout << "In dense callback..." << std::endl;
  num_msgs_ += 1;
  if(!depthmap_)
  {
    ROS_ERROR("depthmap not initialized. Call the DepthmapNode::init() method");
    return;
  }
  cv::Mat img_8uC1;
  try
  {
    cv_bridge::CvImageConstPtr cv_img_ptr =
        cv_bridge::toCvShare(dense_input->image,
                             dense_input,
                             sensor_msgs::image_encodings::MONO8);
    img_8uC1 = cv_img_ptr->image;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
  }
  rmd::SE3<float> T_world_curr(
        dense_input->pose.orientation.w,
        dense_input->pose.orientation.x,
        dense_input->pose.orientation.y,
        dense_input->pose.orientation.z,
        dense_input->pose.position.x,
        dense_input->pose.position.y,
        dense_input->pose.position.z);

  // visualize camera pose
  Matrix<float, 3, 4> cam_pose = T_world_curr.data; // row major
  cv::Mat pose_mat(3, 3, CV_32F);
  float* mat_pointer = (float*)pose_mat.data;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      mat_pointer[3*row + col] = cam_pose(row, col);
    }
  }
  viz_pose_.rotation(pose_mat);
  viz_pose_.translation(
      cv::Vec3f(100*cam_pose(0, 3), 100*cam_pose(1, 3), 100*cam_pose(2, 3)));
  viz_window_.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(50.0));
  viz_window_.setWidgetPose("Dense Input Pose", viz_pose_);
  viz_window_.spinOnce(1, true);

  std::cout << "DEPTHMAP NODE: received image "
            << img_8uC1.cols << "x" << img_8uC1.rows
            <<  std::endl;
  std::cout << "minmax: " << dense_input->min_depth << " " << dense_input->max_depth << std::endl;
  std::cout << "T_world_curr:" << std::endl;
  std::cout << T_world_curr << std::endl;

  switch (state_) {
  case rmd::State::TAKE_REFERENCE_FRAME:
  {
    if(depthmap_->setReferenceImage(
         img_8uC1,
         T_world_curr.inv(),
         dense_input->min_depth,
         dense_input->max_depth))
    {
      state_ = State::UPDATE;
    }
    else
    {
      std::cerr << "ERROR: could not set reference image" << std::endl;
    }
    break;
  }
  case rmd::State::UPDATE:
  {
    depthmap_->update(img_8uC1, T_world_curr.inv());
    const float perc_conv = depthmap_->getConvergedPercentage();
    const float dist_from_ref = depthmap_->getDistFromRef();
    std::cout << "INFO: percentage of converged measurements: " << perc_conv << "%" << std::endl;
    std::cout << "INFO: dist from ref: " << dist_from_ref << std::endl;
    if(perc_conv > ref_compl_perc_ || dist_from_ref > max_dist_from_ref_)
    {
      state_ = State::TAKE_REFERENCE_FRAME;
      denoiseAndPublishResults();
    }
    break;
  }
  default:
    break;
  }
  if(publish_conv_every_n_ < num_msgs_)
  {
    publishConvergenceMap();
    num_msgs_ = 0;
  }
}

void rmd::DepthmapNode::denseInputAndExternalDepthCallback(
    const svo_msgs::DenseInputConstPtr& dense_msg,
    const sensor_msgs::ImageConstPtr& depth_msg)
{
  std::cout << "in combined callback..." << std::endl;
  if (external_depth_uchar_.empty()) {
    external_depth_uchar_.create(cv::Size(depth_msg->width, depth_msg->height), CV_16UC1);
    external_depth_float_.create(cv::Size(depth_msg->width, depth_msg->height), CV_32FC1);
    transformed_external_depth_float_ = cv::Mat::zeros(cv::Size(cam_width_, cam_height_), CV_32FC1);
  }
  if (state_ == rmd::State::TAKE_REFERENCE_FRAME) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    } catch (cv_bridge::Exception& e) {
      ROS_ERROR("depth cv_bridge exception: %s", e.what());
      exit(1);
    }
    cv_ptr->image.copyTo(external_depth_uchar_);
    for (int row = 0; row < external_depth_uchar_.rows; ++row) {
      for (int col = 0; col < external_depth_uchar_.cols; ++col) {
        external_depth_float_.at<float>(row, col) =
            ((float)(external_depth_uchar_.at<unsigned short>(row, col)))/1000.0;
      }
    }

    // Transform to same reference frame
    transformExternalDepthmap();
    external_depth_available_ = true;
    ROS_INFO("Read external depth...");
  }
  denseInputCallback(dense_msg);
}

void rmd::DepthmapNode::transformExternalDepthmap() {

  for (int row = 0; row < external_depth_float_.rows; ++row) {
    for (int col = 0; col < external_depth_float_.cols; ++col) {
      if (external_depth_float_.at<float>(row, col) > 0.0) {
        float x_ext = (col - ext_cx_)*external_depth_float_.at<float>(row, col)/ext_fx_;
        float y_ext = (row - ext_cy_)*external_depth_float_.at<float>(row, col)/ext_fy_;
        float3 point_ext = make_float3(x_ext, y_ext, external_depth_float_.at<float>(row, col));
        float3 point = ext_cam_to_cam_ * point_ext;
        if (point.z > 0.0) {
          int col_cam = (int)((cam_fx_ * point.x / point.z)) + cam_cx_;
          int row_cam = (int)((cam_fy_ * point.y / point.z)) + cam_cy_;
//          std::cout << "row, col: " << row << ", " << col << " " << row_cam << ", " << col_cam << std::endl;
          if (col_cam >= 0 && col_cam < cam_width_ && row_cam >= 0 && row_cam < cam_height_) {
            transformed_external_depth_float_.at<float>(row_cam, col_cam) = point.z;
          }
        }
      }
    }
  }

}

void rmd::DepthmapNode::denoiseAndPublishResults()
{
  depthmap_->downloadDenoisedDepthmap(0.5f, 200);
  depthmap_->downloadConvergenceMap();

  // Get Depthmaps from external source (ex: Kinect/Realsense)
  std::cout << "denoising and publishing.." << std::endl;

  if (external_depth_available_) {
    const cv::Mat depth = depthmap_->getDepthmap();
    cv::Mat_<float> augmented_depth;
    depth.copyTo(augmented_depth);

    // Fuse
    for (int row = 0; row < cam_height_; ++row) {
      for (int col = 0; col < cam_width_; ++col) {
        if (transformed_external_depth_float_.at<float>(row, col) > 0.5 &&
            transformed_external_depth_float_.at<float>(row, col) < 3.0) {
          augmented_depth.at<float>(row, col) = transformed_external_depth_float_.at<float>(row, col);
        }
      }
    }
    std::cout << "here3" << std::endl;

    depthmap_->setAugmentedDepthmap(augmented_depth);

    external_depth_available_ = false;
  }

  std::async(std::launch::async,
             &rmd::Publisher::publishDepthmapAndPointCloud,
             *publisher_);
}

void rmd::DepthmapNode::publishConvergenceMap()
{
  depthmap_->downloadConvergenceMap();

  std::async(std::launch::async,
             &rmd::Publisher::publishConvergenceMap,
             *publisher_);
}
