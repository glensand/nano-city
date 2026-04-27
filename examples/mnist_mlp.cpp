#include "mlp_scalar_static.h"

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
    assert(train_ratio > 0.f && train_ratio < 1.f);
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

int argmax_logits(const std::vector<float>& logits) {
    int best_idx = 0;
    float best_val = logits.front();
    for (int i = 1; i < static_cast<int>(logits.size()); ++i) {
        if (logits[static_cast<std::size_t>(i)] > best_val) {
            best_val = logits[static_cast<std::size_t>(i)];
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
        192);
    const auto [train, test] = split_dataset(full, 0.7f);
    const auto load_end = clock_t::now();
    std::cout << "[trace] mnist_load_ms=" << ms_t(load_end - load_start).count() << '\n';

    scalar_static::mlp net(28 * 28, { 32, 32, 10 });
    constexpr float lr = 0.05f;
    constexpr int epochs = 100;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        const auto epoch_start = clock_t::now();
        float epoch_loss = 0.f;
        int correct = 0;
        double forward_loss_ms = 0.0;
        double backward_ms = 0.0;
        double step_ms = 0.0;

        for (std::size_t i = 0; i < train.images.size(); ++i) {
            const auto forward_start = clock_t::now();
            const auto loss = net.forward_loss(train.images[i], train.labels[i]);
            const auto& logits = net.forward(train.images[i]);
            const auto forward_end = clock_t::now();
            forward_loss_ms += ms_t(forward_end - forward_start).count();

            const auto backward_start = clock_t::now();
            net.set_loss_grad(1.f);
            net.backward();
            const auto backward_end = clock_t::now();
            backward_ms += ms_t(backward_end - backward_start).count();

            const auto step_start = clock_t::now();
            net.step(lr); 
            const auto step_end = clock_t::now();
            step_ms += ms_t(step_end - step_start).count();

            epoch_loss += loss;
            correct += (argmax_logits(logits) == static_cast<int>(train.labels[i])) ? 1 : 0;
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
        if (100.f * static_cast<float>(correct) / static_cast<float>(train.images.size()) > 90.f) {
            break;
        }
    }

    const auto test_eval_start = clock_t::now();
    int test_correct = 0;
    for (std::size_t i = 0; i < test.images.size(); ++i) {
        const auto& logits = net.forward(test.images[i]);
        test_correct += (argmax_logits(logits) == static_cast<int>(test.labels[i])) ? 1 : 0;
    }
    const auto test_eval_end = clock_t::now();

    std::cout << "test_acc="
              << (100.f * static_cast<float>(test_correct) / static_cast<float>(test.images.size()))
              << "% on " << test.images.size() << " samples\n"
              << "[trace] test_eval_ms=" << ms_t(test_eval_end - test_eval_start).count() << '\n';

    std::cout << "\nInteractive validation viewer\n";
    std::cout << "Press Enter/n for next sample, q to quit.\n\n";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    for (std::size_t i = 0; i < test.images.size(); ++i) {
        const auto& logits = net.forward(test.images[i]);
        const int pred = argmax_logits(logits);
        const int target = static_cast<int>(test.labels[i]);

        std::cout << "Sample " << i << "/" << (test.images.size() - 1)
                  << " predicted=" << pred << " target=" << target
                  << (pred == target ? " [OK]\n" : " [MISS]\n");
        print_mnist_ascii(test.images[i]);
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
