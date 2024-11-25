#pragma once

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

class DKD {
public:
    DKD(int radius = 2, int top_k = -1, float scores_th = 0.2, int n_limit = 20000);
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> 
    detect_keypoints(torch::Tensor scores_map, bool sub_pixel = true);
    
    std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>> 
    forward(torch::Tensor scores_map, bool sub_pixel = true);

private:
    int radius_;
    int top_k_;
    float scores_th_;
    int n_limit_;
    int kernel_size_;
    float temperature_;
    torch::nn::Unfold unfold_;
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
           int n_limit = 5000,
           bool load_pretrained = true);

    std::tuple<torch::Tensor, torch::Tensor> extract_dense_map(torch::Tensor image);
    
    torch::Dict<std::string, torch::Tensor> forward(torch::Tensor image);
    
    torch::Dict<std::string, torch::Tensor> run(cv::Mat& img_rgb);

private:
    // Feature extraction layers
    torch::nn::AvgPool2d pool2_{nullptr}, pool4_{nullptr};
    torch::nn::Sequential block1_{nullptr}, block2_{nullptr}, block3_{nullptr}, block4_{nullptr};
    torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr}, conv3_{nullptr}, conv4_{nullptr};
    torch::nn::Sequential score_head_{nullptr};
    
    std::shared_ptr<DKD> dkd_;
    std::shared_ptr<SDDH> desc_head_;
    
    torch::Device device_;
    int dim_;
    
    void init_layers(const std::string& model_name);
    void load_weights(const std::string& model_name);
};
