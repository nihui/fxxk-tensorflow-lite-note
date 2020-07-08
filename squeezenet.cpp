// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "platform.h"
#include "net.h"


void ncnn_debug(ncnn::Mat& ncnn_img)
{
    cv::Mat a(ncnn_img.h, ncnn_img.w, CV_8UC3);
    const float mean_vals[3] = {-1.0f, -1.0f, -1.0f};
    const float normalize_vals[3] = {127.5f, 127.5f, 127.5f};
    ncnn_img.substract_mean_normalize(mean_vals, normalize_vals); 
    ncnn_img.to_pixels(a.data, ncnn::Mat::PIXEL_RGB);

    cv::imwrite("/home/gx/myproj/tensorflow2ncnn/build/examples/output2.png", a);
}


void checkMatElement(ncnn::Mat& ncnn_img,int length){

    float* ptr0 = ncnn_img.channel(0);
    for(int i;i<length;i++){ 
        
        std::cout<<*ptr0<<" ";
        ptr0++;
    }
    std::cout<<std::endl;
}

static int apply_net(const cv::Mat& bgr,const cv::Mat& bgr_mask)
{
    ncnn::Net mynet;

    mynet.load_param("mynet.param");
    mynet.load_model("mynet.bin");


    // process b_i =>(tf)batch_incomplete
    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_RGB, bgr.cols, bgr.rows);
    const float mean_vals[3] = {127.5f,127.5f,127.5f};
    const float normalize_vals[3] = {1/127.5f,1/127.5f,1/127.5f};
    in.substract_mean_normalize(mean_vals, normalize_vals);
    // std::cout<<in.elemsize<<std::endl;
    // checkMatElement(in,20);
    // process mask =>(tf)masks
    ncnn::Mat mask = ncnn::Mat::from_pixels(bgr_mask.data, ncnn::Mat::PIXEL_RGB, bgr.cols, bgr.rows);
    // std::cout<<mask.elemsize<<std::endl;


    



    ncnn::Extractor ex = mynet.create_extractor();

    ex.input("data", in);
    ex.input("mask", mask);

    ncnn::Mat out;
    ex.extract("mask_1", out);
    checkMatElement(out,20);
    ncnn_debug(out);
    
    return 0;
}



using namespace cv;

int main(int argc, char** argv)
{
    // const char* imagepath = argv[1];
    // cv::Mat m = cv::imread(imagepath, 1);

    cv::Mat image = cv::imread("/home/gx/myproj/tensorflow2ncnn/build/examples/case1_raw.png", 1);
    cv::Mat mask = cv::imread("/home/gx/myproj/tensorflow2ncnn/build/examples/case1_mask.png",1);
    cv::Mat mask_b = mask.clone();
    cv::threshold(mask,mask_b,127.5f,1.0f,CV_THRESH_BINARY);

    apply_net(image,mask_b);

    return 0;
}
