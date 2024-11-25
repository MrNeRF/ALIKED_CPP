#include "ALIKED.hpp"
#include "torch/torch.h"

namespace F = torch::nn::functional;

using namespace torch::indexing;

DKD::DKD(int radius, int top_k, float scores_th, int n_limit)
    : radius_(radius),
      top_k_(top_k),
      scores_th_(scores_th),
      n_limit_(n_limit),
      kernel_size_(2 * radius + 1),
      temperature_(0.1),
      unfold_(torch::nn::UnfoldOptions(kernel_size_).padding(radius)) {

    // Initialize the hw_grid
    auto x = torch::linspace(-radius_, radius_, kernel_size_);
    auto meshgrid = torch::meshgrid({x, x});
    hw_grid_ = torch::stack({meshgrid[1], meshgrid[0]}, -1)
                   .reshape({-1, 2}); // (kernel_size*kernel_size) x 2
}

torch::Tensor DKD::simple_nms(const torch::Tensor& scores, int nms_radius) {
    auto zeros = torch::zeros_like(scores);
    auto kernel_size = nms_radius * 2 + 1;

    auto max_pool_options = torch::nn::functional::MaxPool2dFuncOptions(kernel_size)
                                .stride(1)
                                .padding(nms_radius);

    auto max_mask = scores == torch::nn::functional::max_pool2d(scores, max_pool_options);

    for (int i = 0; i < 2; ++i)
    {
        auto supp_mask = torch::nn::functional::max_pool2d(
                             max_mask.to(torch::kFloat), max_pool_options) > 0;
        auto supp_scores = torch::where(supp_mask, zeros, scores);
        auto new_max_mask = supp_scores == torch::nn::functional::max_pool2d(
                                               supp_scores, max_pool_options);
        max_mask = max_mask | (new_max_mask & (~supp_mask));
    }

    return torch::where(max_mask, scores, zeros);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
DKD::detect_keypoints(torch::Tensor scores_map, bool sub_pixel) {
    const auto batch_size = scores_map.size(0);
    const auto height = scores_map.size(2);
    const auto width = scores_map.size(3);

    auto scores_nograd = scores_map.detach();
    auto nms_scores = simple_nms(scores_nograd, 2);

    // Remove border
    nms_scores.index_put_({Slice(), Slice(), Slice(None, radius_), Slice()}, 0);
    nms_scores.index_put_({Slice(), Slice(), Slice(), Slice(None, radius_)}, 0);
    nms_scores.index_put_({Slice(), Slice(), Slice(-radius_, None), Slice()}, 0);
    nms_scores.index_put_({Slice(), Slice(), Slice(), Slice(-radius_, None)}, 0);

    std::vector<torch::Tensor> keypoints;
    std::vector<torch::Tensor> scoredispersitys;
    std::vector<torch::Tensor> kptscores;
    std::vector<torch::Tensor> indices_keypoints;

    // Detect keypoints without grad
    if (top_k_ > 0)
    {
        auto flat_scores = nms_scores.reshape({batch_size, -1});
        auto topk = flat_scores.topk(top_k_);
        for (int64_t i = 0; i < batch_size; ++i)
        {
            indices_keypoints.push_back(std::get<1>(topk)[i]);
        }
    } else
    {
        auto masks = nms_scores > scores_th_;
        if (masks.sum().item<int64_t>() == 0)
        {
            auto th = scores_nograd.reshape({batch_size, -1}).mean(1);
            masks = nms_scores > th.reshape({batch_size, 1, 1, 1});
        }
        masks = masks.reshape({batch_size, -1});

        auto scores_view = scores_nograd.reshape({batch_size, -1});
        for (int64_t i = 0; i < batch_size; ++i)
        {
            auto indices = masks[i].nonzero().squeeze(1);
            if (indices.size(0) > n_limit_)
            {

                auto kpts_sc = scores_view[i].index_select(0, indices);
                auto sort_idx = std::get<1>(kpts_sc.sort(true));
                indices = indices.index_select(0, sort_idx.slice(0, n_limit_));
            }
            indices_keypoints.push_back(indices);
        }
    }

    auto wh = torch::tensor({width - 1.0f, height - 1.0f}, scores_nograd.options());

    if (sub_pixel)
    {
        auto patches = unfold_(scores_map);
        hw_grid_ = hw_grid_.to(scores_map.device());

        for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx)
        {
            auto patch = patches[batch_idx].transpose(0, 1);
            auto indices_kpt = indices_keypoints[batch_idx];
            auto patch_scores = patch.index_select(0, indices_kpt);

            auto keypoints_xy_nms = torch::stack({indices_kpt % width,
                                                  torch::div(indices_kpt, width, /*rounding_mode=*/"floor")},
                                                 1);

            auto max_v_tuple = patch_scores.max(1);
            auto max_v = std::get<0>(max_v_tuple).detach().unsqueeze(1);
            auto x_exp = ((patch_scores - max_v) / temperature_).exp();

            // Soft-argmax
            auto xy_residual = x_exp.mm(hw_grid_) / x_exp.sum(1).unsqueeze(1);

            auto hw_grid_expanded = hw_grid_.unsqueeze(0);
            auto xy_residual_expanded = xy_residual.unsqueeze(1);
            auto dist2 = (hw_grid_expanded - xy_residual_expanded)
                             .div(radius_)
                             .norm(2, -1)
                             .pow(2);

            auto scoredispersity = (x_exp * dist2).sum(1) / x_exp.sum(1);

            auto keypoints_xy = keypoints_xy_nms + xy_residual;
            keypoints_xy = keypoints_xy.div(wh).mul(2).sub(1);

            auto kptscore = torch::nn::functional::grid_sample(
                scores_map[batch_idx].unsqueeze(0),
                keypoints_xy.view({1, 1, -1, 2}),
                torch::nn::functional::GridSampleFuncOptions()
                    .mode(torch::kBilinear)
                    .align_corners(true))[0][0][0];

            keypoints.push_back(keypoints_xy);
            scoredispersitys.push_back(scoredispersity);
            kptscores.push_back(kptscore);
        }
    } else
    {
        for (int64_t batch_idx = 0; batch_idx < batch_size; ++batch_idx)
        {
            auto indices_kpt = indices_keypoints[batch_idx];

            auto keypoints_xy = torch::stack({indices_kpt % width,
                                              torch::div(indices_kpt, width, /*rounding_mode=*/"floor")},
                                             1);

            keypoints_xy = keypoints_xy.div(wh).mul(2).sub(1);

            auto kptscore = torch::nn::functional::grid_sample(
                scores_map[batch_idx].unsqueeze(0),
                keypoints_xy.view({1, 1, -1, 2}),
                torch::nn::functional::GridSampleFuncOptions()
                    .mode(torch::kBilinear)
                    .align_corners(true))[0][0][0];

            keypoints.push_back(keypoints_xy);
            scoredispersitys.push_back(kptscore); // For non-subpixel case
            kptscores.push_back(kptscore);
        }
    }

    return std::make_tuple(keypoints, scoredispersitys, kptscores);
}

std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
DKD::forward(torch::Tensor scores_map, bool sub_pixel) {
    return detect_keypoints(scores_map, sub_pixel);
}