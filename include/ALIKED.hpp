#pragma once

#include "blocks.hpp"
#include "input_padder.hpp"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include <fstream>
#include <map>
#include <memory>
#include <string>

struct AlikedConfig {
    int c1, c2, c3, c4, dim, K, M;
};

const std::map<std::string, AlikedConfig> ALIKED_CFGS = {
    {"aliked-t16", {8, 16, 32, 64, 64, 3, 16}},
    {"aliked-n16", {16, 32, 64, 128, 128, 3, 16}},
    {"aliked-n16rot", {16, 32, 64, 128, 128, 3, 16}},
    {"aliked-n32", {16, 32, 64, 128, 128, 3, 32}}};

class DKD : public torch::nn::Module {
public:
    DKD(int radius = 2, int top_k = -1, float scores_th = 0.2, int n_limit = 20000);
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    detect_keypoints(torch::Tensor scores_map, bool sub_pixel = true);

    torch::Tensor simple_nms(const torch::Tensor& scores, int nms_radius);

    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    forward(torch::Tensor scores_map, bool sub_pixel = true);

private:
    int radius_;
    int top_k_;
    float scores_th_;
    int n_limit_;
    int kernel_size_;
    float temperature_;
    torch::nn::Unfold unfold_{nullptr};
    torch::Tensor hw_grid_;
};

class SDDH : public torch::nn::Module {
public:
    SDDH(int dims, int kernel_size = 3, int n_pos = 8, bool conv2D = false, bool mask = false);
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    forward(torch::Tensor x, std::vector<torch::Tensor>& keypoints);

private:
    int kernel_size_;
    int n_pos_;
    bool conv2D_;
    bool mask_;
    torch::nn::Sequential offset_conv_{nullptr};
    torch::nn::Conv2d sf_conv_{nullptr};
    torch::nn::Conv2d convM_{nullptr};
    torch::Tensor agg_weights_;
};

class ALIKED : public torch::nn::Module {
public:
    ALIKED(const std::string& model_name = "aliked-n32",
           const std::string& device = "cuda",
           int top_k = -1,
           float scores_th = 0.2,
           int n_limit = 5000);

    std::tuple<torch::Tensor, torch::Tensor> extract_dense_map(torch::Tensor image);
    torch::Dict<std::string, torch::Tensor> forward(torch::Tensor image);
    torch::Dict<std::string, torch::Tensor> run(cv::Mat& img_rgb);

private:
    void init_layers(const std::string& model_name);
    void load_weights(const std::string& model_name);

    // Utility functions to import weigths from Python
    void load_parameters(const std::string& pt_pth);
    static std::vector<char> get_the_bytes(const std::string& filename);

    // Feature extraction layers
    torch::nn::AvgPool2d pool2_{nullptr}, pool4_{nullptr};
    std::shared_ptr<torch::nn::Module> block1_;
    std::shared_ptr<torch::nn::Module> block2_;
    std::shared_ptr<torch::nn::Module> block3_;
    std::shared_ptr<torch::nn::Module> block4_;
    torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr},
        conv3_{nullptr}, conv4_{nullptr};
    torch::nn::Sequential score_head_{nullptr};

    std::shared_ptr<DKD> dkd_;
    std::shared_ptr<SDDH> desc_head_;

    torch::Device device_;
    int dim_{};
};