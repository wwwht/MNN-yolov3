//
//  pictureRecognition.cpp
//  MNN
//
//  Created by MNN on 2018/05/14.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"
#include <unordered_map>
#include <tuple>
#include <iomanip>
#define MNN_OPEN_TIME_TRACE

#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
// #include "AutoTime.hpp"
#include <sys/time.h>

#define STB_IMAGE_IMPLEMENTATION

// #include "stb_image.h"
// #include "stb_image_write.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <algorithm>

using namespace MNN;
using namespace MNN::CV;
using std::cin;
using std::endl;
using std::cout;
// class name
// static std::vector<std::string> class_names = {
    // "pedestrian", "people", "bicycle", "car", "van", 
    // "truck", "tricycle", "awning-tricycle","bus", "motor"};
// bbox conrains label and prob
static std::vector<std::string> class_names = {
    "pedestrian", "vehicle"};
// bbox conrains label and prob
struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};
// 按概率排序，降序
static void qsort_descent_inplace(std::vector<Object> &objects, int left, int right) {
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j) {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &objects) {
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}
// 计算相交区域面积
static inline float intersection_area(const Object &a, const Object &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}
// NMS 合并预测的bbox
static void nms_sorted_bboxes(const std::vector<Object> &objects, std::vector<int> &picked, float NMS_THRES) {
    picked.clear();  //消除picked中已经存在的元素

    const int n = objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = objects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const Object &a = objects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const Object &b = objects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
//             float IoU = inter_area / union_area
            if (inter_area / union_area > NMS_THRES)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}
// 画出bbox
static cv::Mat draw_objects(const cv::Mat &rgb, const std::vector<Object> &objects) {

    cv::Mat image = rgb.clone();
    // cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    for (size_t i = 0; i < objects.size(); i++) {
        const Object &obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        // char text[256];
        // sprintf(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);

        // int baseLine = 0;
        // cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        // int x = obj.rect.x;
        // int y = obj.rect.y - label_size.height - baseLine;
        // if (y < 0)
        //     y = 0;
        // if (x + label_size.width > image.cols)
        //     x = image.cols - label_size.width;

        // cv::rectangle(image, cv::Rect(cv::Point(x, y),
        //                               cv::Size(label_size.width, label_size.height + baseLine)),
        //               cv::Scalar(255, 255, 255), -1);

        // cv::putText(image, text, cv::Point(x, y + label_size.height),
        //             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    return image;
}

static void initConfig(MNN::ScheduleConfig &config, MNN::CV::ImageProcess::Config &img_config)
{
    // config.type  = MNN_FORWARD_VULKAN;
    config.type  = MNN_FORWARD_CPU;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_High;
    config.backendConfig = &backendConfig;
    img_config.filterType = MNN::CV::BILINEAR;

    const float mean_vals[3] = {0.456f*255.f, 0.406f*255.f, 0.485f*255.f};
    const float norm_vals[3] = {1/0.224f/255.f, 1/0.225/255.f, 1/0.229/255.f};
    // const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
    // const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
    ::memcpy(img_config.mean, mean_vals, sizeof(mean_vals));
  	::memcpy(img_config.normal, norm_vals, sizeof(norm_vals));
    img_config.sourceFormat = MNN::CV::BGR;
	img_config.destFormat = MNN::CV::RGB;
    img_config.wrap = MNN::CV::CLAMP_TO_EDGE;
}

static uint8_t* imgTrans(const cv::Mat &img_orig)
{
    cv::Mat img;
    cv::resize(img_orig, img, cv::Size(640, 384));
    uchar* imgdata = new uchar[img.total()*4];
    cv::Mat MatTemp(img.size(), CV_8UC4, imgdata);
    cv::cvtColor(img, MatTemp, CV_BGR2RGBA, 4); 
    uint8_t *pImg = (uint8_t*)MatTemp.data;
    return pImg;
    
}

static float sigmoid(float x)
{
    return (1.0 / (1 + exp(-x)));
}
static void topk_(float* input, int k, int length, std::vector<std::tuple<int, float>> &res){
    std::unordered_map<int, float> m;
    //自定义结构的比较器，这里为优先级队列实现一个Great比较器，使优先级队列元素从小到大跑得了排序
    struct cmpPairSecondFloatGreat{
        bool operator() (const std::pair<int, float>&a, const std::pair<int, float>& b) {
            return a.second > b.second; // 按照value建小跟对
        }
    };
    //定义优先级队列，队列实现最小堆，即top为最小值。
    std::priority_queue<std::pair<int, float>, std::vector<std::pair<int, float>>, cmpPairSecondFloatGreat> topk_q;
    // for (auto a : nums) ++m[a];
    for(int i=0;i<length;i++){
        m.insert(std::make_pair(i, *(input + i))); // map(inds, value)
    }
    for(auto mm:m){
        topk_q.push(std::make_pair(mm.first, mm.second)); // map(inds, value)
        if(topk_q.size() > k){
            topk_q.pop();
        }
    }
    for(int i=0;i<k;i++){
        res.push_back(std::make_tuple(topk_q.top().first, topk_q.top().second));
        topk_q.pop();
    }
} 

void MaxPool2d(const float* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,float* top_data, std::vector<std::tuple<int, int, int>> &point)
{
    const int w = width;
    const int h = height;
    const int m = channels;
    const int n = num;
    const int d = pooled_height; // kernel size
    const int zh = height - 2;
    const int zw = width - 2;
    std::cout << zh << "and " << zw << std::endl;
    std::cout << w << "and " << h << std::endl;
    int i,j,k,o,u,v,index,index2=0;
    std::tuple<int, int, int>tmp;
    float s;
    for (o = 0; o < m; ++o)
        for (i = 0; i < zh; ++i)
            for (j = 0; j < zw; ++j)
            {
                index = o*h*w+i*w+j;
                s=-10000.0;
                // for (u = 0; u < d&&(u+d*i)<h; ++u)
                    // for (v = 0; v < d&&(v+d*j)<w; ++v)
                for (u = 0; u < d; ++u)
                    for (v = 0; v < d; ++v)
                        // std::cout << *(bottom_data+index+u*w+v) << std::endl;
                        if (sigmoid(*(bottom_data+index+u*w+v))>s){
                            s=sigmoid(*(bottom_data+index+u*w+v));
                            tmp = std::make_tuple(o, i+v, j+u);
                        }
                *(top_data+index2)=s;
                point.push_back(tmp);
                // std::cout << std::get<0>(tmp) << ", " << std::get<1>(tmp) << ", " << std::get<2>(tmp) << std::endl;
                ++index2;
            }
}

int main(int argc, const char *argv[]) {
    if (argc < 3) {
        MNN_PRINT("Usage: ./yolo.out model.mnn input.jpg [word.txt]\n");
        return 0;
    }
    std::string mode = "centernet";
    // std::string mode = "yolo";

    const float NMS_THRES = 0.6f;
    const float CONF_THRES = 0.2f;
    const int num_category=int(class_names.size());

    timeval startime, endtime;
    int pad_W, pad_H;
    double ratio;
    cv::Mat raw_image;
    
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(argv[1]));
    ScheduleConfig config;
    config.numThread = 1;
    // config.type = MNN_FORWARD_AUTO;
    config.type = MNN_FORWARD_CPU;
    auto session = net->createSession(config);
    auto input = net->getSessionInput(session, NULL);
    std::vector<int> shape = input->shape();
    std::cout << shape[0] << ", " << shape[1] << ", "<< shape[2] << ", " << shape[3] << std::endl;
    // int input_H=shape[1];
    // int input_W=shape[2];
    // net->resizeTensor(input, shape);
    // net->resizeSession(session);

    MNN::CV::ImageProcess::Config img_config;
    initConfig(config, img_config);
    MNN::CV::ImageProcess* imgprocess = MNN::CV::ImageProcess::create(img_config); // 
    // MNN::Session* session = net->createSession(config); // session
    auto inputPatch = argv[2];
    raw_image = cv::imread(inputPatch);
    std::cout <<raw_image.size() << std::endl;
    // cv::cvtColor(raw_image, raw_image, cv::COLOR_BGR2RGB);
    cv::Mat img;
    cv::resize(raw_image, img, cv::Size(800, 800));

    // auto output = net->getSessionOutput(session, NULL);
    // std::vector<int> shape2 = output->shape();
    // std::cout << shape2[0] << ", " << shape2[1] << ", "<< shape2[2] << ", " << shape2[3] << std::endl;
    // MNN::Tensor input_tensor(input, input->getDimensionType());
    auto input_tensor = new MNN::Tensor(input, input->getDimensionType());
    // MNN::Tensor output_host(output, output->getDimensionType());

    // uint8_t *pImg = imgTrans(frame);
    // imgprocess->convert(pImg, 640, 384, 0, input_tensor);
    imgprocess->convert((uint8_t*)img.data, 800, 800, 0, input_tensor);
    input->copyFromHostTensor(input_tensor);

    // //Image Preprocessing
    // {
    //     gettimeofday(&startime, nullptr);
    //     auto inputPatch = argv[2];
    //     raw_image = cv::imread(inputPatch);
    //     cv::cvtColor(raw_image, raw_image, cv::COLOR_BGR2RGB);

    //     int ori_height = raw_image.rows;
    //     int ori_width = raw_image.cols;
    //     ratio = std::min(1.0 * input_H / ori_height, 1.0 * input_W / ori_width);
    //     int resize_height = int(ori_height * ratio);
    //     int resize_width = int(ori_width * ratio);
    //     //odd number->pad size error
    //     if (resize_height%2!=0) resize_height-=1;
    //     if (resize_width%2!=0) resize_width-=1;

    //     pad_W = int((input_W - resize_width) / 2);
    //     pad_H = int((input_H - resize_height) / 2);
    //     cv::Scalar pad(128, 128, 128);
    //     cv::Mat resized_image;
    //     // cv::resize(raw_image, resized_image, cv::Size(resize_width, resize_height), 0, 0, cv::INTER_LINEAR);
    //     cv::resize(raw_image, resized_image, cv::Size(800, 800), 0, 0, cv::INTER_LINEAR);
    //     // cv::copyMakeBorder(resized_image, resized_image, pad_H, pad_H, pad_W, pad_W, cv::BORDER_CONSTANT, pad);
    //     resized_image.convertTo(resized_image, CV_32FC3);
    //     resized_image = resized_image / 255.0f;
    //     // wrapping input tensor, convert nhwc to nchw
    //     auto nchw_Tensor = new Tensor(input, MNN::Tensor::CAFFE); 
    //     // std::vector<int> dim{1, input_H, input_W, 3}; // 创建tensorflow格式
    //     // auto nhwc_Tensor = MNN::Tensor::create<float>(dim, NULL, MNN::Tensor::TENSORFLOW);  // 创建nhwc格式tensor
    //     auto nhwc_data = nchw_Tensor->host<float>();
    //     auto nhwc_size = nchw_Tensor->size();
    //     ::memcpy(nhwc_data, resized_image.data, nhwc_size);
    //     input->copyFromHostTensor(nchw_Tensor);
    //     gettimeofday(&endtime, nullptr);
    //     cout << "preprocesstime: " << (endtime.tv_sec-startime.tv_sec)*1000+(endtime.tv_usec - startime.tv_usec) / 1000 << "ms" << endl;
    // }
    //Image Inference

    {
        gettimeofday(&startime, nullptr);
        net->runSession(session);
        gettimeofday(&endtime, nullptr);
        cout << "inferencetime: " << (endtime.tv_sec-startime.tv_sec)*1000+(endtime.tv_usec - startime.tv_usec) / 1000 << "ms" << endl;

    }
    //Image PostProcess
    if(mode == "yolo")
    {
        gettimeofday(&startime, nullptr);
        auto output = net->getSessionOutput(session, NULL);
        std::vector<int> shape2 = output->shape();
        std::cout << shape2[0] << ", " << shape2[1] << ", "<< shape2[2] << ", " << shape2[3] << std::endl;

        auto dimType = output->getDimensionType();
        if (output->getType().code != halide_type_float) {
            dimType = Tensor::TENSORFLOW;
        }
        std::shared_ptr<Tensor> outputUser(new Tensor(output, dimType));
        output->copyToHostTensor(outputUser.get());
        auto type = outputUser->getType();

        auto size = outputUser->elementSize();
        std::vector<float> tempValues(size);
        if (type.code == halide_type_float) {
            auto values = outputUser->host<float>();
            for (int i = 0; i < size; ++i) {
                tempValues[i] = values[i];
            }
        }

        auto OUTPUT_NUM = outputUser->shape()[0];
        std::vector<std::vector<Object> > class_candidates(20);
        std::vector<int> tempcls;

        for (int i = 0; i < OUTPUT_NUM; ++i) {
            auto prob = tempValues[i * (5+num_category) + 4];
            auto maxcls = std::max_element(tempValues.begin() + i * (5+num_category) + 5, tempValues.begin() + i * (5+num_category) + (5+num_category));
            auto clsidx = maxcls - (tempValues.begin() + i * (5+num_category) + 5);
            auto score = prob * (*maxcls);
            if (score < CONF_THRES) continue;
            auto xmin = (tempValues[i * (5+num_category) + 0] - pad_W) / ratio;
            auto xmax = (tempValues[i * (5+num_category) + 2] - pad_W) / ratio;
            auto ymin = (tempValues[i * (5+num_category) + 1] - pad_H) / ratio;
            auto ymax = (tempValues[i * (5+num_category) + 3] - pad_H) / ratio;

            Object obj;
            obj.rect = cv::Rect_<float>(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
            obj.label = clsidx;
            obj.prob = score;
            class_candidates[clsidx].push_back(obj);
        }
        std::vector<Object> objects;
        for (int i = 0; i < (int) class_candidates.size(); i++) {
            std::vector<Object> &candidates = class_candidates[i];

            qsort_descent_inplace(candidates);

            std::vector<int> picked;
            nms_sorted_bboxes(candidates, picked, NMS_THRES);

            for (int j = 0; j < (int) picked.size(); j++) {
                int z = picked[j];
                objects.push_back(candidates[z]);
            }
        }
        gettimeofday(&endtime, nullptr);
        cout << "postprocesstime: " << (endtime.tv_sec-startime.tv_sec)*1000+(endtime.tv_usec - startime.tv_usec) / 1000 << "ms" << endl;
        auto imgshow = draw_objects(raw_image, objects);
        cv::imwrite("./test.png", imgshow);
        // cv::imshow("w", imgshow);
        // cv::waitKey(-1);
        return 0;
    }
    if(mode == "centernet"){
        gettimeofday(&startime, nullptr);
        // std::string output_tensor_name0 = "602";
        // std::string output_tensor_name1 = "603";
        // std::string output_tensor_name2 = "604";
        std::string output_tensor_name0 = "534";
        std::string output_tensor_name1 = "535";
        std::string output_tensor_name2 = "536";
        // auto output = net->getSessionOutput(session, NULL);
        // auto outputTensor = net->getSessionOutput(session, NULL);

        MNN::Tensor *tensor_scores  = net->getSessionOutput(session, output_tensor_name0.c_str());
        MNN::Tensor *tensor_boxes   = net->getSessionOutput(session, output_tensor_name1.c_str());
        MNN::Tensor *tensor_anchors = net->getSessionOutput(session, output_tensor_name2.c_str());
        std::vector<int> shape2 = tensor_scores->shape();
        std::cout << shape2[0] << ", " << shape2[1] << ", "<< shape2[2] << ", " << shape2[3] << std::endl;
        std::vector<int> shape3 = tensor_boxes->shape();
        std::cout << shape3[0] << ", " << shape3[1] << ", "<< shape3[2] << ", " << shape3[3] << std::endl;
        std::vector<int> shape4 = tensor_anchors->shape();
        std::cout << shape4[0] << ", " << shape4[1] << ", "<< shape4[2] << ", " << shape4[3] << std::endl;

        MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
        MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
        MNN::Tensor tensor_anchors_host(tensor_anchors, tensor_anchors->getDimensionType());

        tensor_scores->copyToHostTensor(&tensor_scores_host);
        tensor_boxes->copyToHostTensor(&tensor_boxes_host);
        tensor_anchors->copyToHostTensor(&tensor_anchors_host);

        std::cout << "batch:    " << tensor_scores->batch()    << std::endl 
              << "channels: " << tensor_scores->channel()  << std::endl
              << "height:   " << tensor_scores->height()   << std::endl
              << "width:    " << tensor_scores->width()    << std::endl
              << "type:     " << tensor_scores->getDimensionType() << std::endl;

        // post processing steps
        auto scores_dataPtr  = tensor_scores_host.host<float>();
        auto boxes_dataPtr   = tensor_boxes_host.host<float>();
        auto anchors_dataPtr = tensor_anchors_host.host<float>();

        // 存储到TXT
        std::ofstream outfile;
        outfile.open("/home/user/work/ckpt_103.txt", std::ios::app);
        for(int i=0;i<2;i++){
            for(int j=0;j<200;j++){
                for(int k=0;k<200;k++){
                    int index = i*200*200+j*200+k;
                    outfile << std::setprecision(5) << scores_dataPtr[index] << " " ;
                }
            }
        }

        // std::cout << scores_dataPtr[0]<< std::endl;
        // std::cout << boxes_dataPtr[0] << std::endl;
        // std::cout << anchors_dataPtr[0] << std::endl;

        // 1. heatmap 进行 sigmoid
        // std::vector<std::vector<Object> > class_candidates(10);
        // int cha = tensor_scores->channel();
        // int hei = tensor_scores->height();
        // int wid = tensor_scores->width();
        // int out_h = wid - 2;
        // // std::vector<float> temp(cha*hei*wid,0);
        // // std::vector<std::vector<float>> tmp(cha, std::vector<float>(hei*wid));
        // float *output = new float [cha*out_h*out_h*sizeof(float)]; // maxpool out put
        // // float *inds = new float [cha*out_h*out_h*sizeof(float)]; // 存放max值得index
        // // 定义一个二维数组out[]， 输出topk的位置坐标
        // // heatmap 长度数组，sigmoid后滑动窗口
        // // 2. 滑动窗口NMS
        // std::vector<std::tuple<int,int,int>>points;
        // std::vector<int>bbox_class;
        // std::vector<std::tuple<int, float>>res;
        // std::vector<cv::Point>point_inds, topk_points;
        // // float* test = new float[5*5*sizeof(float)];
        // // for(int i=0;i<25;i++){
        //     // *(test+i) = i;
        // // }

        // MaxPool2d(scores_dataPtr,1,cha,hei,wid,3,output,points);
        // // float *output = new float [1*3*3*sizeof(float)];
        // // MaxPool2d(test,1,1,5,5,3,output,points);
        // // for(int i=0;i<9;i++){
        // //     std::cout << *(output+i) << std::endl;
        // // }
        // topk_(output, 100, cha *out_h * out_h, res);
        // // for(int i =0;i<100;i++){
        // //     std::cout << "index 和值是: "<<std::get<0>(res[i]) << ", " << std::get<1>(res[i]) << std::endl;
        // //     std::cout << std::get<0>(points[std::get<0>(res[i])])<< std::endl;
        // //     std::cout << std::get<1>(points[std::get<0>(res[i])])<<  "**"<<std::get<2>(points[std::get<0>(res[i])])<<std::endl;
        // // }
        // // 3. 找到topk 的heapmap位置和wh，regs：res中存储map(inds, value),根据inds找到regs和wh中的值
        // std::vector<cv::Point>regs;
        // std::vector<cv::Point>wh;
        // std::vector<cv::Point>top_left, bottom_right;
        // std::vector<Object> bbox;

        // for(int i=0;i<res.size();i++){ // 遍历找到的topk
        //     int o = std::get<0>(points[std::get<0>(res[i])]);
        //     int x = std::get<1>(points[std::get<0>(res[i])]);
        //     int y = std::get<2>(points[std::get<0>(res[i])]);
        //     // std::cout << o << ", " << x << ", " << y << std::endl;
        //     bbox_class.push_back(o); // label信息
        //     cv::Point tmp = cv::Point(y, x);
        //     // cv::Point reg = cv::Point(y * wid + x, o*wid*hei + y*wid+x);
        //     cv::Point w_h_ = cv::Point(*(anchors_dataPtr + wid * hei + y * wid + x), *(anchors_dataPtr + y*wid+x));
        //     std::cout << tmp.x << ", " <<tmp.y << " wh is : "<< w_h_.x << ", " << w_h_.y << std::endl;
        //     // 将point进行融合
        //     cv::Point bottom = (tmp + w_h_ / 2) * 4;
        //     cv::Point top ;
        //     top.x = (tmp.x - w_h_.x / 2) * 4;
        //     top.y = (tmp.y - w_h_.y / 2) * 4;
        //     w_h_.x = w_h_.x * 4;
        //     w_h_.y = w_h_.y * 4;
        //     // top_left.push_back(top);
        //     // bottom_right.push_back(bottom); // 拿到左上和右下的点
        //     Object obj;
        //     obj.rect = cv::Rect(top.x, top.y, w_h_.y, w_h_.x);
        //     obj.label = o;
        //     obj.prob = std::get<1>(res[i]);
        //     class_candidates[o].push_back(obj);

        //     // bbox.push_back(obj);
        // }
        // std::vector<Object> objects;
        // for (int i = 0; i < (int) class_candidates.size(); i++) {
        //     std::vector<Object> &candidates = class_candidates[i];

        //     qsort_descent_inplace(candidates);

        //     std::vector<int> picked;
        //     nms_sorted_bboxes(candidates, picked, NMS_THRES);

        //     for (int j = 0; j < (int) picked.size(); j++) {
        //         int z = picked[j];
        //         objects.push_back(candidates[z]);
        //     }
        // }
        // auto imgshow = draw_objects(img, objects);
        // cv::imwrite("./test_1123_3.png", imgshow);
        // 4. 输出位置和bbox的长宽
        

        // for (int i = 0; i < size; ++i) {
        //     tempValues[i] = heat_map[i];
        // }
        // auto OUTPUT_NUM = outputUser->shape()[0];
        // std::cout << OUTPUT_NUM << std::endl;
    }
}
