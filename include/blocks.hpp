#pragma once
#include <torch/torch.h>

class ConvBlock : public torch::nn::Module {
public:
    ConvBlock(int in_channels, int out_channels,
              const std::string& conv_type = "conv",
              bool mask = false);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr};
    torch::nn::BatchNorm2d bn1_{nullptr}, bn2_{nullptr};
};

class ResBlock : public torch::nn::Module {
public:
    ResBlock(int inplanes, int planes, int stride = 1,
             std::shared_ptr<torch::nn::Module> downsample = nullptr,
             const std::string& conv_type = "conv",
             bool mask = false);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr};
    torch::nn::BatchNorm2d bn1_{nullptr}, bn2_{nullptr};
    std::shared_ptr<torch::nn::Module> downsample_;
    int stride_;
};