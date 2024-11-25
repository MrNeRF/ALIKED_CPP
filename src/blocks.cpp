#include "blocks.hpp"
#include "deform_conv2d.h"

torch::nn::Conv2d get_conv(int inplanes, int planes, int kernel_size = 3,
                           int stride = 1, int padding = 1, bool bias = false,
                           const std::string& conv_type = "conv", bool mask = false) {
    if (conv_type == "conv")
    {
        return torch::nn::Conv2d(
            torch::nn::Conv2dOptions(inplanes, planes, kernel_size)
                .stride(stride)
                .padding(padding)
                .bias(bias));
    } else if (conv_type == "dcn")
    {
        // Note: For now, using regular conv as DCN requires additional implementation
        return torch::nn::Conv2d(
            torch::nn::Conv2dOptions(inplanes, planes, kernel_size)
                .stride(stride)
                .padding(padding)
                .bias(bias));
    } else
    {
        throw std::runtime_error("Unknown conv type: " + conv_type);
    }
}

ConvBlock::ConvBlock(int in_channels, int out_channels,
                     const std::string& conv_type, bool mask) {
    conv1_ = register_module("conv1",
                             get_conv(in_channels, out_channels, 3, 1, 1, false, conv_type, mask));
    bn1_ = register_module("bn1",
                           torch::nn::BatchNorm2d(out_channels));

    conv2_ = register_module("conv2",
                             get_conv(out_channels, out_channels, 3, 1, 1, false, conv_type, mask));
    bn2_ = register_module("bn2",
                           torch::nn::BatchNorm2d(out_channels));
}

torch::Tensor ConvBlock::forward(torch::Tensor x) {
    x = torch::selu(bn1_->forward(conv1_->forward(x)));
    x = torch::selu(bn2_->forward(conv2_->forward(x)));
    return x;
}

ResBlock::ResBlock(int inplanes, int planes, int stride,
                   std::shared_ptr<torch::nn::Module> downsample,
                   const std::string& conv_type, bool mask)
    : downsample_(downsample),
      stride_(stride) {

    conv1_ = register_module("conv1",
                             get_conv(inplanes, planes, 3, 1, 1, false, conv_type, mask));
    bn1_ = register_module("bn1",
                           torch::nn::BatchNorm2d(planes));

    conv2_ = register_module("conv2",
                             get_conv(planes, planes, 3, 1, 1, false, conv_type, mask));
    bn2_ = register_module("bn2",
                           torch::nn::BatchNorm2d(planes));

    if (downsample)
    {
        register_module("downsample", downsample);
    }
}

torch::Tensor ResBlock::forward(torch::Tensor x) {
    auto identity = x;

    x = conv1_->forward(x);
    x = bn1_->forward(x);
    x = torch::selu(x);

    x = conv2_->forward(x);
    x = bn2_->forward(x);

    if (downsample_)
    {
        identity = downsample_->as<torch::nn::Conv2d>()->forward(identity);
    }

    x += identity;
    x = torch::selu(x);

    return x;
}

DeformableConv2d::DeformableConv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
                                   bool bias, bool mask) {

    padding_ = padding;
    mask_ = mask;
    kernel_size_ = kernel_size;

    // 3 * kernel_size * kernel_size if mask else 2 * kernel_size * kernel_size
    int channel_num = mask ? 3 * kernel_size * kernel_size : 2 * kernel_size * kernel_size;

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

    if (mask_) {
        // Split into offset and mask
        auto chunks = out.chunk(3, 1);
        auto o1 = chunks[0];
        auto o2 = chunks[1];
        mask = torch::sigmoid(chunks[2]);
        offset = torch::cat({o1, o2}, 1);
    } else {
        offset = out;
        mask = torch::Tensor();
    }

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
            mask_
    );
}
