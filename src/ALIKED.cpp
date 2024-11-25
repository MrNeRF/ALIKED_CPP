#include "ALIKED.hpp"

namespace F = torch::nn::functional;

struct AlikedConfig {
    int c1, c2, c3, c4, dim, K, M;
};

static std::map<std::string, AlikedConfig> ALIKED_CFGS = {
    {"aliked-t16", {8, 16, 32, 64, 64, 3, 16}},
    {"aliked-n16", {16, 32, 64, 128, 128, 3, 16}},
    {"aliked-n16rot", {16, 32, 64, 128, 128, 3, 16}},
    {"aliked-n32", {16, 32, 64, 128, 128, 3, 32}}
};

ALIKED::ALIKED(const std::string& model_name,
               const std::string& device,
               int top_k,
               float scores_th,
               int n_limit,
               bool load_pretrained)
    : device_(torch::Device(device)) {
    
    init_layers(model_name);
    
    dkd_ = std::make_shared<DKD>(2, top_k, scores_th, n_limit);
    
    if (load_pretrained) {
        load_weights(model_name);
    }
    
    this->to(device_);
    this->eval();
}

void ALIKED::init_layers(const std::string& model_name) {
    auto config = ALIKED_CFGS[model_name];
    dim_ = config.dim;
    
    // Initialize pooling layers
    pool2_ = register_module("pool2", 
        torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2)));
    pool4_ = register_module("pool4",
        torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(4).stride(4)));
    
    // Initialize blocks
    auto gate = torch::nn::SELU();
    auto norm = torch::nn::BatchNorm2d;
    
    // Block implementations...
    // (Similar structure to Python, using ConvBlock and ResBlock)
    
    // Initialize conv layers
    conv1_ = register_module("conv1", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(config.c1, dim_ / 4, 1)));
    // ... similarly for conv2_, conv3_, conv4_
    
    // Initialize score head
    torch::nn::Sequential score_head;
    score_head->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(dim_, 8, 1)));
    score_head->push_back(torch::nn::SELU());
    // ... add remaining layers
    register_module("score_head", score_head);
    
    // Initialize descriptor head
    desc_head_ = std::make_shared<SDDH>(dim_, config.K, config.M);
}

std::tuple<torch::Tensor, torch::Tensor> 
ALIKED::extract_dense_map(torch::Tensor image) {
    // Implementation similar to Python version
    auto x1 = block1_->forward(image);
    auto x2 = block2_->forward(pool2_->forward(x1));
    auto x3 = block3_->forward(pool4_->forward(x2));
    auto x4 = block4_->forward(pool4_->forward(x3));
    
    // Feature aggregation
    x1 = F::selu(conv1_->forward(x1));
    x2 = F::selu(conv2_->forward(x2));
    x3 = F::selu(conv3_->forward(x3));
    x4 = F::selu(conv4_->forward(x4));
    
    // Upsample and concatenate
    auto x2_up = F::interpolate(x2, 
        F::InterpolateFuncOptions().scale_factor(std::vector<double>{2.0})
            .mode(torch::kBilinear).align_corners(true));
    // ... similarly for x3_up and x4_up
    
    auto x1234 = torch::cat({x1, x2_up, x3_up, x4_up}, 1);
    
    auto score_map = torch::sigmoid(score_head_->forward(x1234));
    auto feature_map = F::normalize(x1234, F::NormalizeFuncOptions().p(2).dim(1));
    
    return std::make_tuple(feature_map, score_map);
}

torch::Dict<std::string, torch::Tensor>
ALIKED::forward(torch::Tensor image) {
    auto [feature_map, score_map] = extract_dense_map(image);
    auto [keypoints, kptscores, scoredispersitys] = dkd_->forward(score_map);
    auto [descriptors, offsets] = desc_head_->forward(feature_map, keypoints);
    
    torch::Dict<std::string, torch::Tensor> output;
    output.insert("keypoints", keypoints[0]);
    output.insert("descriptors", descriptors[0]);
    output.insert("scores", kptscores[0]);
    output.insert("score_dispersity", scoredispersitys[0]);
    output.insert("score_map", score_map);
    
    return output;
}

torch::Dict<std::string, torch::Tensor>
ALIKED::run(cv::Mat& img_rgb) {
    cv::Mat float_img;
    img_rgb.convertTo(float_img, CV_32F, 1.0/255.0);
    
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device_);
    
    std::vector<torch::Tensor> tensor_channels;
    for(const auto& channel : channels) {
        tensor_channels.push_back(
            torch::from_blob(channel.data, {channel.rows, channel.cols}, options));
    }
    
    auto img_tensor = torch::stack(tensor_channels, 0)
                         .unsqueeze(0);
    
    auto pred = forward(img_tensor);
    
    auto kpts = pred["keypoints"];
    auto h = float_img.rows;
    auto w = float_img.cols;
    auto wh = torch::tensor({w - 1, h - 1}, kpts.options());
    kpts = wh * (kpts + 1) / 2;
    
    pred["keypoints"] = kpts;
    
    return pred;
}
