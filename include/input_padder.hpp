#pragma once
#include <torch/torch.h>

class InputPadder {
public:
    InputPadder(int h, int w, int div_by = 8)
        : ht_(h),
          wd_(w) {
        int pad_ht = (((ht_ / div_by) + 1) * div_by - ht_) % div_by;
        int pad_wd = (((wd_ / div_by) + 1) * div_by - wd_) % div_by;

        pad_ = {pad_wd / 2, pad_wd - pad_wd / 2,
                pad_ht / 2, pad_ht - pad_ht / 2};
    }

    torch::Tensor pad(const torch::Tensor& x) {
        return torch::nn::functional::pad(
            x,
            torch::nn::functional::PadFuncOptions({pad_[0], pad_[1], pad_[2], pad_[3]})
                .mode(torch::kReplicate));
    }

    torch::Tensor unpad(const torch::Tensor& x) {
        int h = x.size(-2);
        int w = x.size(-1);
        return x.index({torch::indexing::Slice(),
                        torch::indexing::Slice(),
                        torch::indexing::Slice(pad_[2], h - pad_[3]),
                        torch::indexing::Slice(pad_[0], w - pad_[1])});
    }

private:
    int ht_;
    int wd_;
    std::vector<int> pad_;
};