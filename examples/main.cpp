#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include "ALIKED.hpp"

namespace fs = std::filesystem;

class ImageLoader {
public:
    explicit ImageLoader(const std::string& filepath) {
        for (const auto& entry : fs::directory_iterator(filepath)) {
            const auto& path = entry.path();
            std::string ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            
            if (ext == ".png" || ext == ".jpg" || ext == ".ppm") {
                images_.push_back(path.string());
            }
        }
        std::sort(images_.begin(), images_.end());
        std::cout << "Loading " << images_.size() << " images" << std::endl;
    }
    
    cv::Mat operator[](size_t idx) const {
        return cv::imread(images_[idx]);
    }
    
    size_t size() const { return images_.size(); }
    
private:
    std::vector<std::string> images_;
};

torch::Tensor mnn_matcher(const torch::Tensor& desc1, const torch::Tensor& desc2) {
    auto sim = torch::matmul(desc1, desc2.transpose(0, 1));
    sim.masked_fill_(sim < 0.75, 0);
    
    auto nn12 = std::get<1>(sim.max(1));
    auto nn21 = std::get<1>(sim.max(0));
    
    auto ids1 = torch::arange(sim.size(0), sim.options());
    auto mask = ids1 == nn21.index({nn12});
    
    auto matches_ids1 = torch::masked_select(ids1, mask);
    auto matches_ids2 = torch::masked_select(nn12, mask);
    
    return torch::stack({matches_ids1, matches_ids2}, 1);
}

cv::Mat plot_keypoints(const cv::Mat& image, const torch::Tensor& kpts,
                      int radius = 2, const cv::Scalar& color = cv::Scalar(0, 0, 255)) {
    cv::Mat out = image.clone();
    
    auto kpts_a = kpts.round().to(torch::kInt64).cpu();
    auto kpts_accessor = kpts_a.accessor<int64_t, 2>();
    
    for (int i = 0; i < kpts_a.size(0); i++) {
        cv::Point center(kpts_accessor[i][0], kpts_accessor[i][1]);
        cv::circle(out, center, radius, color, -1, cv::LINE_4);
    }
    return out;
}

cv::Mat plot_matches(const cv::Mat& image0, const cv::Mat& image1,
                    const torch::Tensor& kpts0, const torch::Tensor& kpts1,
                    const torch::Tensor& matches,
                    int radius = 2,
                    const cv::Scalar& color = cv::Scalar(255, 0, 0),
                    const cv::Scalar& mcolor = cv::Scalar(0, 255, 0)) {
    cv::Mat out0 = plot_keypoints(image0, kpts0, radius, color);
    cv::Mat out1 = plot_keypoints(image1, kpts1, radius, color);
    
    int H0 = image0.rows, W0 = image0.cols;
    int H1 = image1.rows, W1 = image1.cols;
    int H = std::max(H0, H1);
    int W = W0 + W1;
    
    cv::Mat out(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    out0.copyTo(out(cv::Rect(0, 0, W0, H0)));
    out1.copyTo(out(cv::Rect(W0, 0, W1, H1)));
    
    auto matches_a = matches.cpu();
    auto kpts0_a = kpts0.cpu();
    auto kpts1_a = kpts1.cpu();
    auto matches_accessor = matches_a.accessor<int64_t, 2>();
    
    for (int i = 0; i < matches_a.size(0); i++) {
        int idx0 = matches_accessor[i][0];
        int idx1 = matches_accessor[i][1];
        
        cv::Point2f pt0(kpts0_a[idx0][0].item<float>(), kpts0_a[idx0][1].item<float>());
        cv::Point2f pt1(kpts1_a[idx1][0].item<float>() + W0, kpts1_a[idx1][1].item<float>());
        
        cv::line(out, pt0, pt1, mcolor, 1, cv::LINE_AA);
    }
    
    return out;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_dir> [options]" << std::endl;
        return 1;
    }
    
    // Parse command line arguments
    std::string input_dir = argv[1];
    std::string model_name = "aliked-n32";
    std::string device = "cuda";
    int top_k = -1;
    float scores_th = 0.2f;
    int n_limit = 5000;
    
    // Initialize model
    std::cout << "Initializing ALIKED model..." << std::endl;
    ALIKED model(model_name, device, top_k, scores_th, n_limit);
    
    // Load images
    ImageLoader image_loader(input_dir);
    if (image_loader.size() < 2) {
        std::cerr << "Need at least 2 images in the input directory" << std::endl;
        return 1;
    }
    
    // Process reference image
    cv::Mat img_ref = image_loader[0];
    cv::Mat img_rgb_ref;
    cv::cvtColor(img_ref, img_rgb_ref, cv::COLOR_BGR2RGB);
    
    auto pred_ref = model.run(img_rgb_ref);
    auto kpts_ref = pred_ref.at("keypoints");
    auto desc_ref = pred_ref.at("descriptors");

    std::cout << "Press 'space' to start. \nPress 'q' or 'ESC' to stop!" << std::endl;
    
    for (size_t i = 1; i < image_loader.size(); i++) {
        cv::Mat img = image_loader[i];
        if (img.empty()) break;
        
        cv::Mat img_rgb;
        cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
        
        auto pred = model.run(img_rgb);
        auto kpts = pred.at("keypoints");
        auto desc = pred.at("descriptors");

        auto matches = mnn_matcher(desc_ref, desc);
        
        std::string status = "matches/keypoints: " + 
                            std::to_string(matches.size(0)) + "/" + 
                            std::to_string(kpts.size(0));
        
        cv::Mat vis_img = plot_matches(img_ref, img, kpts_ref, kpts, matches);
        
        cv::putText(vis_img, "Press 'q' or 'ESC' to stop.", 
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, 
                    cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        
        cv::namedWindow(model_name);
        cv::setWindowTitle(model_name, model_name + ": " + status);
        cv::imshow(model_name, vis_img);
        
        char c = cv::waitKey(0);
        if (c == 'q' || c == 27) break;
    }
    
    std::cout << "Finished!" << std::endl;
    std::cout << "Press any key to exit!" << std::endl;
    
    cv::destroyAllWindows();
    return 0;
}
