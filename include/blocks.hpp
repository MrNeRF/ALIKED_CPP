#pragma once
#include <torch/torch.h>
#include <memory>

class DeformableConv2d : public torch::nn::Module {
public:
    DeformableConv2d(int in_channels, int out_channels,
                     int kernel_size = 3, int stride = 1,
                     int padding = 1, bool bias = false, bool mask = false);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d offset_conv_{nullptr};
    torch::nn::Conv2d regular_conv_{nullptr};
    int padding_;
    bool mask_;
    int kernel_size_;
    int groups_ = 1;
    int mask_offset_ = 0;
};

class ConvBlock : public torch::nn::Module {
public:
    ConvBlock(int in_channels, int out_channels,
              const std::string& conv_type = "conv",
              bool mask = false);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr};
    std::shared_ptr<DeformableConv2d> deform1_{nullptr}, deform2_{nullptr};
    torch::nn::BatchNorm2d bn1_{nullptr}, bn2_{nullptr};
};

class ResBlock : public torch::nn::Module {
public:
    ResBlock(int inplanes, int planes, int stride = 1,
             const std::shared_ptr<torch::nn::Module>& downsample = nullptr,
             const std::string& conv_type = "conv",
             bool mask = false);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv2d conv1_{nullptr}, conv2_{nullptr};
    std::shared_ptr<DeformableConv2d> deform1_{nullptr}, deform2_{nullptr};
    torch::nn::BatchNorm2d bn1_{nullptr}, bn2_{nullptr};
    std::shared_ptr<torch::nn::Module> downsample_;
};