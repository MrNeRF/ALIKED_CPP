#include "blocks.hpp"
#include "deform_conv2d.h"

ConvBlock::ConvBlock(int in_channels, int out_channels,
                     const std::string& conv_type, bool mask) {

    if (conv_type == "conv") {

        auto conv1 = torch::nn::Conv2d((torch::nn::Conv2dOptions(in_channels, out_channels, 3)
                    .stride(1)
                    .padding(1)
                    .bias(false)));
        conv1_ = register_module("conv1", conv1 );

        auto conv2 = torch::nn::Conv2d((torch::nn::Conv2dOptions(out_channels, out_channels, 3)
                    .stride(1)
                    .padding(1)
                    .bias(false)));
        conv2_ = register_module("conv2", conv2);

    } else {
        auto conv1 = std::make_shared<DeformableConv2d>(
                in_channels,
                out_channels,
                3,
                1,
                1,
                false);
        deform1_ = register_module("conv1", conv1);

        auto conv2 = std::make_shared<DeformableConv2d>(
                out_channels,
                out_channels,
                3,
                1,
                1,
                false);
        deform2_ = register_module("conv2", conv2);
    }

    bn1_ = register_module("bn1", torch::nn::BatchNorm2d(out_channels));
    bn2_ = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
}

torch::Tensor ConvBlock::forward(torch::Tensor x) {
    if (conv1_ && conv2_) {
        x = torch::selu(bn1_->forward(conv1_->forward(x)));
        x = torch::selu(bn2_->forward(conv2_->forward(x)));
    } else {
        x = torch::selu(bn1_->forward(deform1_->forward(x)));
        x = torch::selu(bn2_->forward(deform2_->forward(x)));
    }
    return x;
}

ResBlock::ResBlock(int inplanes, int planes, int stride,
                   const torch::nn::Conv2d& downsample,
                   const std::string& conv_type)
    : downsample_(downsample) {

    if (conv_type == "conv") {

        auto conv1 = torch::nn::Conv2d((torch::nn::Conv2dOptions(inplanes, planes, 3)
                .stride(stride)
                .padding(1)
                .bias(false)));
        conv1_ = register_module("conv1", conv1 );

        auto conv2 = torch::nn::Conv2d((torch::nn::Conv2dOptions(planes, planes, 3)
                .stride(stride)
                .padding(1)
                .bias(false)));
        conv2_ = register_module("conv2", conv2);

    } else {
        auto conv1 = std::make_shared<DeformableConv2d>(
                inplanes,
                planes,
                3,
                1,
                1,
                false);
        deform1_ = register_module("conv1", conv1);

        auto conv2 = std::make_shared<DeformableConv2d>(
                planes,
                planes,
                3,
                1,
                1,
                false);
        deform2_ = register_module("conv2", conv2);
    }

    bn1_ = register_module("bn1",
                           torch::nn::BatchNorm2d(planes));
    bn2_ = register_module("bn2",
                           torch::nn::BatchNorm2d(planes));

    if (downsample)
    {
        register_module("downsample", downsample);
    }
}

torch::Tensor ResBlock::forward(torch::Tensor x) {
    auto identity = x;

    if (conv1_ && conv2_) {
        x = conv1_->forward(x);
        x = bn1_->forward(x);
        x = torch::selu(x);

        x = conv2_->forward(x);
        x = bn2_->forward(x);
    } else {
        x = deform1_->forward(x);
        x = bn1_->forward(x);
        x = torch::selu(x);

        x = deform2_->forward(x);
        x = bn2_->forward(x);
    }
    if (downsample_)
    {
        identity = downsample_->as<torch::nn::Conv2d>()->forward(identity);
    }

    x += identity;
    x = torch::selu(x);

    return x;
}

DeformableConv2d::DeformableConv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                                   bool bias) {

    padding_ = padding;
    const int channel_num = 2 * kernel_size * kernel_size;

    // Register offset conv
    offset_conv_ = register_module("offset_conv",
                                   torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, channel_num, kernel_size)
                                                             .stride(stride)
                                                             .padding(padding)
                                                             .bias(true)));

    // Register regular conv
    regular_conv_ = register_module("regular_conv",
                                    torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                                                              .stride(stride)
                                                              .padding(padding)
                                                              .bias(bias)));
}

torch::Tensor DeformableConv2d::forward(torch::Tensor x) {
    auto h = x.size(2);
    auto w = x.size(3);
    float max_offset = std::max(h, w) / 4.0f;

    auto out = offset_conv_->forward(x);
    torch::Tensor offset, mask;

    offset = out;
    mask = torch::Tensor();

    // Clamp offset values
    offset = offset.clamp(-max_offset, max_offset);

    // Torchvision's deform_conv2d
    return vision::ops::deform_conv2d(
            x,                          // input
            regular_conv_->weight,      // weight
            offset,
            mask,
            regular_conv_->bias,        // bias
            1,1,// stride
            padding_, padding_, // padding
            1, 1, // dilation
            groups_,                    // groups
            mask_offset_              , // mask
           false
    );
}
