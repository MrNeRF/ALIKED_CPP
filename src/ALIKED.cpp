#include "ALIKED.hpp"

#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

ALIKED::ALIKED(const std::string& model_name,
               const std::string& device,
               int top_k,
               float scores_th,
               int n_limit)
    : device_(torch::Device(device)),
    dim_{-1}
    {


    // Initialize DKD and descriptor head
    dkd_ = std::make_shared<DKD>(2, top_k, scores_th, n_limit);

    auto config = ALIKED_CFGS.at(model_name);
    desc_head_ = std::make_shared<SDDH>(config.dim, config.K, config.M);

    // Initialize layers
    init_layers(model_name);
    load_weights(model_name);

    this->to(device_);
    this->eval();
}

void ALIKED::init_layers(const std::string& model_name) {
    auto config = ALIKED_CFGS.at(model_name);
    dim_ = config.dim;

    // Basic layers first
    pool2_ = register_module("pool2", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2)));
    pool4_ = register_module("pool4", torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(4).stride(4)));

    // Block 1
    auto block1 = std::make_shared<ConvBlock>(3, config.c1, "conv", false);
    register_module("block1", block1);

    // Block 2
    auto downsample2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(config.c1, config.c2, 1));
    auto block2 = std::make_shared<ResBlock>(config.c1, config.c2, 1,downsample2,
                                             "conv");
    register_module("block2", block2);

    // Block 3
    auto downsample3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(config.c2, config.c3, 1));
    auto block3 = std::make_shared<ResBlock>(config.c2, config.c3, 1,downsample3,
                                             "dcn");
    register_module("block3", block3);

    // Block 4
    auto downsample4 = torch::nn::Conv2d(torch::nn::Conv2dOptions(config.c3, config.c4, 1));
    auto block4 = std::make_shared<ResBlock>(config.c3, config.c4, 1, downsample4,
                                             "dcn");
    register_module("block4", block4);

    // Convolution layers
    conv1_ = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(config.c1, dim_ / 4, 1)));
    conv2_ = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(config.c2, dim_ / 4, 1)));
    conv3_ = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(config.c3, dim_ / 4, 1)));
    conv4_ = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(config.c4, dim_ / 4, 1)));

    // Score head
    auto score_head = torch::nn::Sequential();
    score_head->push_back("0", torch::nn::Conv2d(torch::nn::Conv2dOptions(dim_, 8, 1)));
    score_head->push_back("1", torch::nn::SELU());
    score_head->push_back("2", torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 4, 3).padding(1)));
    score_head->push_back("3", torch::nn::SELU());
    score_head->push_back("4", torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 4, 3).padding(1)));
    score_head->push_back("5", torch::nn::SELU());
    score_head->push_back("6", torch::nn::Conv2d(torch::nn::Conv2dOptions(4, 1, 3).padding(1)));
    register_module("score_head", score_head);

    // Descriptor head
    register_module("desc_head", desc_head_);
    // DKD
    register_module("dkd", dkd_);
}

std::tuple<torch::Tensor, torch::Tensor>
ALIKED::extract_dense_map(torch::Tensor image) {
    // Create padder for input
    auto padder = InputPadder(image.size(2), image.size(3), 32);
    image = padder.pad(image);

    // Feature extraction
    auto x1 = block1_->forward(image);
    auto x2 = block2_->forward(pool2_->forward(x1));
    auto x3 = block3_->forward(pool4_->forward(x2));
    auto x4 = block4_->forward(pool4_->forward(x3));

    // Feature aggregation
    x1 = torch::selu(conv1_->forward(x1));
    x2 = torch::selu(conv2_->forward(x2));
    x3 = torch::selu(conv3_->forward(x3));
    x4 = torch::selu(conv4_->forward(x4));

    // Upsample
    auto options = torch::nn::functional::InterpolateFuncOptions()
                       .mode(torch::kBilinear)
                       .align_corners(true);

    auto x2_up = torch::nn::functional::interpolate(x2,
                                                    options.size(std::vector<int64_t>{x1.size(2), x1.size(3)}));
    auto x3_up = torch::nn::functional::interpolate(x3,
                                                    options.size(std::vector<int64_t>{x1.size(2), x1.size(3)}));
    auto x4_up = torch::nn::functional::interpolate(x4,
                                                    options.size(std::vector<int64_t>{x1.size(2), x1.size(3)}));

    auto x1234 = torch::cat({x1, x2_up, x3_up, x4_up}, 1);

    // Generate score map and feature map
    auto score_map = torch::sigmoid(score_head_->forward(x1234));
    auto feature_map = torch::nn::functional::normalize(x1234,
                                                        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    // Unpad
    feature_map = padder.unpad(feature_map);
    score_map = padder.unpad(score_map);

    return std::make_tuple(feature_map, score_map);
}

torch::Dict<std::string, torch::Tensor>
ALIKED::forward(torch::Tensor image) {
    auto start_time = std::chrono::high_resolution_clock::now();

    auto [feature_map, score_map] = extract_dense_map(image);
    auto [keypoints, kptscores, scoredispersitys] = dkd_->forward(score_map);
    auto [descriptors, offsets] = desc_head_->forward(feature_map, keypoints);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count() /
                    1000.0f;

    torch::Dict<std::string, torch::Tensor> output;
    output.insert("keypoints", keypoints[0]);
    output.insert("descriptors", descriptors[0]);
    output.insert("scores", kptscores[0]);
    output.insert("score_dispersity", scoredispersitys[0]);
    output.insert("score_map", score_map);
    output.insert("time", torch::tensor(duration));

    return output;
}

torch::Dict<std::string, torch::Tensor>
ALIKED::run(cv::Mat& img_rgb) {
    // Convert OpenCV image to torch tensor
    cv::Mat float_img;
    img_rgb.convertTo(float_img, CV_32F, 1.0 / 255.0);

    // Split channels and create tensor
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(device_);

    std::vector<torch::Tensor> tensor_channels;
    for (const auto& channel : channels)
    {
        tensor_channels.push_back(
            torch::from_blob(channel.data, {channel.rows, channel.cols}, options).clone());
    }

    auto img_tensor = torch::stack(tensor_channels, 0)
                          .unsqueeze(0)
                          .to(device_);

    // Forward pass
    auto pred = forward(img_tensor);

    // Convert keypoints from normalized coordinates to image coordinates
    auto kpts = pred.at("keypoints");
    auto h = float_img.rows;
    auto w = float_img.cols;
    auto wh = torch::tensor({w - 1.0f, h - 1.0f}, kpts.options());
    kpts = wh * (kpts + 1) / 2;

    pred.insert("keypoints", kpts);

    return pred;
}

void ALIKED::load_weights(const std::string& model_name) {
    std::string model_path = ("../models/" + (model_name + ".pt"));

    if (!fs::exists(model_path))
    {
        throw std::runtime_error("Cannot find pretrained model: " + model_path);
    }

    std::cout << "Loading " << model_path << std::endl;

    try
    {
        // Load the PyTorch saved model
        //torch::serialize::InputArchive archive;
        //archive.load_from(model_path);
        load_parameters(model_path);
        // Load into the model
        //torch::NoGradGuard no_grad;
        //for (const auto& key : archive.keys())
        //{
        //    torch::Tensor tensor;
        //    archive.read(key, tensor);

        //    // Some PyTorch state dict keys might need to be adjusted for C++
        //    std::string cpp_key = key;
        //    if (key.substr(0, 7) == "module.")
        //    {
        //        cpp_key = key.substr(7); // Remove "module." prefix if exists
        //    }

        //    try
        //    {
        //        auto* param = this->named_parameters(true).find(cpp_key);
        //        if (param != nullptr)
        //        {
        //            param->copy_(tensor);
        //        } else
        //        {
        //            auto* buffer = this->named_buffers(true).find(cpp_key);
        //            if (buffer != nullptr)
        //            {
        //                buffer->copy_(tensor);
        //            } else
        //            {
        //                std::cout << "Warning: Parameter " << cpp_key << " not found in model" << std::endl;
        //            }
        //        }
        //    } catch (const std::exception& e)
        //    {
        //        std::cout << "Error loading parameter " << cpp_key << ": " << e.what() << std::endl;
        //    }
        //}
    } catch (const std::exception& e)
    {
        throw std::runtime_error("Error loading model: " + std::string(e.what()));
    }
}

std::vector<char> ALIKED::get_the_bytes(const std::string &filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

void ALIKED::load_parameters(const std::string &pt_pth) {
    std::vector<char> f = ALIKED::get_the_bytes(pt_pth);
    c10::Dict<at::IValue, at::IValue> weights = torch::pickle_load(f).toGenericDict();

    auto model_params = named_parameters();
    auto model_buffers = named_buffers();

    // Collect parameter names
    std::unordered_map<std::string, torch::Tensor> param_map;
    for (const auto& p : model_params) {
        param_map[p.key()] = p.value();
    }

    // Collect buffer names
    std::unordered_map<std::string, torch::Tensor> buffer_map;
    for (const auto& b : model_buffers) {
        buffer_map[b.key()] = b.value();
    }

    // Update parameters and buffers
    torch::NoGradGuard no_grad;
    for (const auto& w : weights) {
        const std::string name = w.key().toStringRef();
        const at::Tensor param = w.value().toTensor();

        if (param_map.count(name)) {
            if (param_map[name].sizes() == param.sizes()) {
                param_map[name].copy_(param);
            } else {
                std::cerr << "Shape mismatch for parameter: " << name << std::endl;
            }
        } else if (buffer_map.count(name)) {
            if (buffer_map[name].sizes() == param.sizes()) {
                buffer_map[name].copy_(param);
            } else {
                std::cerr << "Shape mismatch for buffer: " << name << std::endl;
            }
        } else {
            std::cerr << name << " does not exist among model parameters or buffers." << std::endl;
        }
    }
}
