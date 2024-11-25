#include "ALIKED.hpp"
#include <torch/torch.h>
#include "get_patches.hpp"

using namespace torch::indexing;

SDDH::SDDH(int dims, int kernel_size, int n_pos, bool conv2D, bool mask)
    : kernel_size_(kernel_size),
      n_pos_(n_pos),
      conv2D_(conv2D),
      mask_(mask) {

    // Channel num for offsets
    int channel_num = mask ? 3 * n_pos : 2 * n_pos;

    // Build offset convolution layers
    torch::nn::Sequential offset_conv;
    offset_conv->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(dims, channel_num, kernel_size)
            .stride(1)
            .padding(0)
            .bias(true)));
    offset_conv->push_back(torch::nn::SELU());
    offset_conv->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(channel_num, channel_num, 1)
            .stride(1)
            .padding(0)
            .bias(true)));

    register_module("offset_conv", offset_conv);
    offset_conv_ = offset_conv;

    // Sampled feature convolution
    sf_conv_ = register_module("sf_conv",
                               torch::nn::Conv2d(torch::nn::Conv2dOptions(dims, dims, 1)
                                                     .stride(1)
                                                     .padding(0)
                                                     .bias(false)));

    if (!conv2D)
    {
        // Register deformable desc weights
        agg_weights_ = register_parameter("agg_weights",
                                          torch::randn({n_pos, dims, dims}));
    } else
    {
        // Register convM
        convM_ = register_module("convM",
                                 torch::nn::Conv2d(torch::nn::Conv2dOptions(dims * n_pos, dims, 1)
                                                       .stride(1)
                                                       .padding(0)
                                                       .bias(false)));
    }
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
SDDH::forward(torch::Tensor x, std::vector<torch::Tensor>& keypoints) {
    const auto batch_size = x.size(0);
    const auto channels = x.size(1);
    const auto height = x.size(2);
    const auto width = x.size(3);

    const auto wh = torch::tensor({width - 1.0f, height - 1.0f}, x.options());
    const float max_offset = std::max(height, width) / 4.0f;

    std::vector<torch::Tensor> offsets;
    std::vector<torch::Tensor> descriptors;

    for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx)
    {
        auto xi = x[batch_idx];
        auto kptsi = keypoints[batch_idx];
        auto kptsi_wh = (kptsi / 2 + 0.5) * wh;
        auto num_keypoints = kptsi_wh.size(0);

        torch::Tensor patch;
        if (kernel_size_ > 1)
        {
            // Get patches using custom op
            auto kptsi_wh_long = kptsi_wh.to(torch::kLong);
            patch = custom_ops::get_patches_forward(xi, kptsi_wh_long, static_cast<long>(kernel_size_));
        } else
        {
            auto kptsi_wh_long = kptsi_wh.to(torch::kLong);
            patch = xi.index({Slice(),
                              kptsi_wh_long.index({Slice(), 1}),
                              kptsi_wh_long.index({Slice(), 0})})
                        .transpose(0, 1)
                        .reshape({num_keypoints, channels, 1, 1});
        }

        // Compute offsets
        auto offset = offset_conv_->forward(patch).clamp(-max_offset, max_offset);

        torch::Tensor mask_weight;
        if (mask_)
        {
            offset = offset.index({Slice(), Slice(), 0, 0})
                         .view({num_keypoints, 3, n_pos_})
                         .permute({0, 2, 1});
            auto offset_xy = offset.index({Slice(), Slice(), Slice(None, 2)});
            mask_weight = torch::sigmoid(offset.index({Slice(), Slice(), 2}));
            offset = offset_xy;
        } else
        {
            offset = offset.index({Slice(), Slice(), 0, 0})
                         .view({num_keypoints, 2, n_pos_})
                         .permute({0, 2, 1});
        }

        offsets.push_back(offset);

        // Sample features at offset positions
        auto pos = kptsi_wh.unsqueeze(1) + offset;
        pos = 2.0 * pos / wh - 1;
        pos = pos.reshape({1, num_keypoints * n_pos_, 1, 2});

        auto features = torch::nn::functional::grid_sample(
            xi.unsqueeze(0), pos,
            torch::nn::functional::GridSampleFuncOptions()
                .mode(torch::kBilinear)
                .align_corners(true));

        features = features.reshape({channels, num_keypoints, n_pos_, 1})
                       .permute({1, 0, 2, 3});

        if (mask_)
        {
            features = features * mask_weight.unsqueeze(1).unsqueeze(-1);
        }

        features = torch::selu(sf_conv_->forward(features)).squeeze(-1);

        torch::Tensor descs;
        if (!conv2D_)
        {
            descs = torch::einsum("ncp,pcd->nd", {features, agg_weights_});
        } else
        {
            features = features.reshape({num_keypoints, -1}).unsqueeze(-1).unsqueeze(-1);
            descs = convM_->forward(features).squeeze();
        }

        // Normalize descriptors
        descs = torch::nn::functional::normalize(descs,
                                                 torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
        descriptors.push_back(descs);
    }

    return std::make_tuple(descriptors, offsets);
}