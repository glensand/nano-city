#include "mlp_tensor_autograd.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct mnist_data {
    std::vector<std::vector<float>> images;
    std::vector<std::uint8_t> labels;
};

std::uint32_t read_be_u32(std::ifstream& in) {
    unsigned char bytes[4]{};
    in.read(reinterpret_cast<char*>(bytes), 4);
    if (!in) {
        throw std::runtime_error("Failed to read u32 from IDX file");
    }
    return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
           (static_cast<std::uint32_t>(bytes[1]) << 16U) |
           (static_cast<std::uint32_t>(bytes[2]) << 8U) |
           static_cast<std::uint32_t>(bytes[3]);
}

mnist_data load_mnist(
    const std::string& images_path,
    const std::string& labels_path,
    const std::size_t max_samples)
{
    std::ifstream images_in(images_path, std::ios::binary);
    std::ifstream labels_in(labels_path, std::ios::binary);
    if (!images_in) {
        throw std::runtime_error("Cannot open MNIST images: " + images_path);
    }
    if (!labels_in) {
        throw std::runtime_error("Cannot open MNIST labels: " + labels_path);
    }

    const auto image_magic = read_be_u32(images_in);
    const auto num_images = read_be_u32(images_in);
    const auto rows = read_be_u32(images_in);
    const auto cols = read_be_u32(images_in);
    if (image_magic != 2051U) {
        throw std::runtime_error("Invalid images IDX magic");
    }

    const auto label_magic = read_be_u32(labels_in);
    const auto num_labels = read_be_u32(labels_in);
    if (label_magic != 2049U) {
        throw std::runtime_error("Invalid labels IDX magic");
    }
    if (num_images != num_labels) {
        throw std::runtime_error("MNIST images/labels count mismatch");
    }

    const std::size_t sample_count = std::min<std::size_t>(max_samples, num_images);
    const std::size_t image_size = static_cast<std::size_t>(rows * cols);

    mnist_data data;
    data.images.resize(sample_count, std::vector<float>(image_size));
    data.labels.resize(sample_count);

    std::vector<unsigned char> image_buffer(image_size);
    for (std::size_t i = 0; i < sample_count; ++i) {
        images_in.read(reinterpret_cast<char*>(image_buffer.data()), static_cast<std::streamsize>(image_size));
        labels_in.read(reinterpret_cast<char*>(&data.labels[i]), 1);
        if (!images_in || !labels_in) {
            throw std::runtime_error("Unexpected EOF while reading MNIST");
        }
        for (std::size_t p = 0; p < image_size; ++p) {
            data.images[i][p] = static_cast<float>(image_buffer[p]) / 255.f;
        }
    }

    return data;
}

std::pair<mnist_data, mnist_data> split_dataset(const mnist_data& full, const float train_ratio) {
    const std::size_t total = full.images.size();
    const std::size_t train_count = static_cast<std::size_t>(static_cast<float>(total) * train_ratio);

    mnist_data train;
    mnist_data eval;
    train.images.reserve(train_count);
    train.labels.reserve(train_count);
    eval.images.reserve(total - train_count);
    eval.labels.reserve(total - train_count);

    for (std::size_t i = 0; i < total; ++i) {
        if (i < train_count) {
            train.images.push_back(full.images[i]);
            train.labels.push_back(full.labels[i]);
        } else {
            eval.images.push_back(full.images[i]);
            eval.labels.push_back(full.labels[i]);
        }
    }
    return { std::move(train), std::move(eval) };
}

std::vector<float> one_hot(const std::uint8_t label, const std::size_t classes) {
    std::vector<float> out(classes, 0.f);
    out[static_cast<std::size_t>(label)] = 1.f;
    return out;
}

int argmax_row(const tensor_autograd::tensor& t) {
    const auto& vals = t.values();
    int best_idx = 0;
    float best_val = vals.front();
    for (int i = 1; i < static_cast<int>(vals.size()); ++i) {
        if (vals[static_cast<std::size_t>(i)] > best_val) {
            best_val = vals[static_cast<std::size_t>(i)];
            best_idx = i;
        }
    }
    return best_idx;
}

void print_mnist_ascii(const std::vector<float>& image) {
    static constexpr int rows = 28;
    static constexpr int cols = 28;
    static constexpr const char* shades = " .:-=+*#%@";

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const float v = image[static_cast<std::size_t>(r * cols + c)];
            const int idx = std::clamp(static_cast<int>(v * 9.f), 0, 9);
            std::cout << shades[idx];
        }
        std::cout << '\n';
    }
}

} // namespace

int main() {
    using clock_t = std::chrono::steady_clock;
    using ms_t = std::chrono::duration<double, std::milli>;

    const std::string base = "data/mnist/";
    const auto load_start = clock_t::now();
    const auto full = load_mnist(
        base + "train-images-idx3-ubyte",
        base + "train-labels-idx1-ubyte",
        2000);
    const auto load_end = clock_t::now();
    std::cout << "[trace] mnist_load_ms=" << ms_t(load_end - load_start).count() << '\n';

    const auto [train, eval] = split_dataset(full, 0.7f);

    tensor_autograd::mlp net(28 * 28, { 32, 32, 10 });
    constexpr float lr = 0.02f;
    constexpr int epochs = 30;

    const auto prep_start = clock_t::now();
    std::vector<tensor_autograd::tensor> train_x;
    std::vector<tensor_autograd::tensor> train_y;
    std::vector<tensor_autograd::tensor> eval_x;
    train_x.reserve(train.images.size());
    train_y.reserve(train.images.size());
    eval_x.reserve(eval.images.size());

    for (std::size_t i = 0; i < train.images.size(); ++i) {
        train_x.push_back(tensor_autograd::tensor::from_data(1, train.images[i].size(), train.images[i], false));
        train_y.push_back(tensor_autograd::tensor::from_data(1, 10, one_hot(train.labels[i], 10), false));
    }
    for (std::size_t i = 0; i < eval.images.size(); ++i) {
        eval_x.push_back(tensor_autograd::tensor::from_data(1, eval.images[i].size(), eval.images[i], false));
    }
    const auto prep_end = clock_t::now();
    std::cout << "[trace] input_prepare_ms=" << ms_t(prep_end - prep_start).count() << '\n';

    for (int epoch = 0; epoch < epochs; ++epoch) {
        const auto epoch_start = clock_t::now();
        float epoch_loss = 0.f;
        int correct = 0;
        double forward_loss_ms = 0.0;
        double backward_ms = 0.0;
        double step_ms = 0.0;

        for (std::size_t i = 0; i < train.images.size(); ++i) {
            const auto& x = train_x[i];
            const auto& y = train_y[i];

            const auto forward_start = clock_t::now();
            const auto logits = net.forward(x);
            const auto loss = (logits - y).square().mean();
            const auto forward_end = clock_t::now();
            forward_loss_ms += ms_t(forward_end - forward_start).count();

            const auto backward_start = clock_t::now();
            net.zero_grad();
            loss.backward();
            const auto backward_end = clock_t::now();
            backward_ms += ms_t(backward_end - backward_start).count();

            const auto step_start = clock_t::now();
            net.step(lr);
            const auto step_end = clock_t::now();
            step_ms += ms_t(step_end - step_start).count();

            epoch_loss += loss.values().front();
            correct += (argmax_row(logits) == static_cast<int>(train.labels[i])) ? 1 : 0;
        }
        const auto epoch_end = clock_t::now();

        std::cout << "epoch=" << epoch
                  << " train_loss=" << (epoch_loss / static_cast<float>(train.images.size()))
                  << " train_acc=" << (100.f * static_cast<float>(correct) / static_cast<float>(train.images.size()))
                  << "%\n"
                  << "[trace] epoch=" << epoch
                  << " total_ms=" << ms_t(epoch_end - epoch_start).count()
                  << " forward_loss_ms=" << forward_loss_ms
                  << " backward_ms=" << backward_ms
                  << " step_ms=" << step_ms
                  << '\n';
    }

    const auto eval_start = clock_t::now();
    int eval_correct = 0;
    for (std::size_t i = 0; i < eval.images.size(); ++i) {
        const auto logits = net.forward(eval_x[i]);
        eval_correct += (argmax_row(logits) == static_cast<int>(eval.labels[i])) ? 1 : 0;
    }
    const auto eval_end = clock_t::now();

    std::cout << "eval_acc="
              << (100.f * static_cast<float>(eval_correct) / static_cast<float>(eval.images.size()))
              << "% on " << eval.images.size() << " samples\n"
              << "[trace] eval_ms=" << ms_t(eval_end - eval_start).count() << '\n';

    const std::string viz_dir = "mnist_tensor_autograd_viz";
    net.export_layer_visualizations(viz_dir);
    std::cout << "Wrote layer visualizations (PPM) to " << viz_dir << "/ (layer_*_weights.ppm, layer_*_bias.ppm)\n";

    std::cout << "\nInteractive validation viewer\n";
    std::cout << "Press Enter/n for next sample, q to quit.\n\n";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    for (std::size_t i = 0; i < eval.images.size(); ++i) {
        const auto logits = net.forward(eval_x[i]);
        const int pred = argmax_row(logits);
        const int target = static_cast<int>(eval.labels[i]);

        std::cout << "Sample " << i << "/" << (eval.images.size() - 1)
                  << " predicted=" << pred << " target=" << target
                  << (pred == target ? " [OK]\n" : " [MISS]\n");
        print_mnist_ascii(eval.images[i]);
        std::cout << "\nnext> ";

        std::string cmd;
        if (!std::getline(std::cin, cmd)) {
            break;
        }
        if (!cmd.empty() && (cmd[0] == 'q' || cmd[0] == 'Q')) {
            break;
        }
    }

    return 0;
}
