#pragma once

#include "grad_tensor_autograd.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace tensor_autograd {

class linear final {
public:
    linear(const std::size_t in_features, const std::size_t out_features)
        : m_weight(tensor::random_uniform(
            in_features,
            out_features,
            1.f / std::sqrt(static_cast<float>(in_features)),
            true))
        , m_bias(1, out_features, 0.f, true)
    {
    }

    [[nodiscard]] tensor forward(const tensor& x) const {
        return x.matmul(m_weight).add_rowwise(m_bias);
    }

    [[nodiscard]] std::vector<tensor*> parameters() {
        return { &m_weight, &m_bias };
    }

    [[nodiscard]] const tensor& weight() const { return m_weight; }
    [[nodiscard]] const tensor& bias() const { return m_bias; }

private:
    tensor m_weight;
    tensor m_bias;
};

class mlp final {
public:
    mlp(const std::size_t nin, const std::vector<std::size_t>& nouts) {
        assert(!nouts.empty());
        std::size_t in = nin;
        m_layers.reserve(nouts.size());
        for (const auto out : nouts) {
            m_layers.emplace_back(in, out);
            in = out;
        }
    }

    [[nodiscard]] tensor forward(const tensor& x) const {
        tensor out = x;
        for (std::size_t i = 0; i < m_layers.size(); ++i) {
            out = m_layers[i].forward(out);
            if (i + 1 != m_layers.size()) {
                out = out.relu();
            }
        }
        return out;
    }

    void zero_grad() {
        for (auto& layer : m_layers) {
            for (auto* p : layer.parameters()) {
                p->zero_grad();
            }
        }
    }

    void step(const float lr) {
        for (auto& layer : m_layers) {
            for (auto* p : layer.parameters()) {
                p->step(lr);
            }
        }
    }

    [[nodiscard]] std::size_t num_layers() const { return m_layers.size(); }
    [[nodiscard]] const linear& layer(const std::size_t index) const { return m_layers[index]; }

    /// Writes PPM images for each linear layer: weight montage (one tile per output neuron)
    /// and an upscaled bias strip. Creates `output_directory` if missing.
    void export_layer_visualizations(const std::string& output_directory) const;

private:
    std::vector<linear> m_layers;
};

inline void mlp::export_layer_visualizations(const std::string& output_directory) const {
    namespace fs = std::filesystem;
    const fs::path out_dir(output_directory);
    fs::create_directories(out_dir);

    const auto write_ppm_p6 = [](const fs::path& path, const int width, const int height,
                                  const std::vector<std::uint8_t>& rgb) {
        std::ofstream f(path, std::ios::binary);
        f << "P6\n" << width << ' ' << height << "\n255\n";
        f.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    };

    const auto tile_shape = [](const std::size_t n) -> std::pair<std::size_t, std::size_t> {
        if (n == 0) {
            return { 1, 1 };
        }
        const auto root = static_cast<std::size_t>(std::ceil(std::sqrt(static_cast<double>(n))));
        const auto h = root;
        const auto w = (n + h - 1) / h;
        return { h, w };
    };

    constexpr int k_scale = 4;

    for (std::size_t li = 0; li < m_layers.size(); ++li) {
        const auto& W = m_layers[li].weight();
        const auto& B = m_layers[li].bias();
        const std::size_t in_f = W.rows();
        const std::size_t out_f = W.cols();
        const auto& vals = W.values();

        float w_min = vals.front();
        float w_max = vals.front();
        for (const float v : vals) {
            w_min = std::min(w_min, v);
            w_max = std::max(w_max, v);
        }
        const float w_span = w_max - w_min;
        const auto norm_w = [&](const float v) -> std::uint8_t {
            if (!(w_span > 0.f)) {
                return 128;
            }
            const float t = (v - w_min) / w_span;
            return static_cast<std::uint8_t>(std::clamp(t * 255.f, 0.f, 255.f));
        };

        const auto [tile_h, tile_w] = tile_shape(in_f);
        const std::size_t tiles_x = static_cast<std::size_t>(
            std::ceil(std::sqrt(static_cast<double>(out_f))));
        const std::size_t tiles_y = (out_f + tiles_x - 1) / tiles_x;

        const int mw = static_cast<int>(tiles_x * tile_w * static_cast<std::size_t>(k_scale));
        const int mh = static_cast<int>(tiles_y * tile_h * static_cast<std::size_t>(k_scale));
        std::vector<std::uint8_t> montage(static_cast<std::size_t>(mw) * static_cast<std::size_t>(mh) * 3, 0);

        for (std::size_t oc = 0; oc < out_f; ++oc) {
            const std::size_t tx = oc % tiles_x;
            const std::size_t ty = oc / tiles_x;
            for (std::size_t tr = 0; tr < tile_h; ++tr) {
                for (std::size_t tc = 0; tc < tile_w; ++tc) {
                    const std::size_t idx = tr * tile_w + tc;
                    const float v = (idx < in_f) ? vals[idx * out_f + oc] : 0.f;
                    const std::uint8_t g = norm_w(v);
                    const int y0 = static_cast<int>(ty * tile_h * static_cast<std::size_t>(k_scale) + tr * static_cast<std::size_t>(k_scale));
                    const int x0 = static_cast<int>(tx * tile_w * static_cast<std::size_t>(k_scale) + tc * static_cast<std::size_t>(k_scale));
                    for (int dy = 0; dy < k_scale; ++dy) {
                        for (int dx = 0; dx < k_scale; ++dx) {
                            const int py = y0 + dy;
                            const int px = x0 + dx;
                            if (py < 0 || px < 0 || py >= mh || px >= mw) {
                                continue;
                            }
                            const std::size_t p = (static_cast<std::size_t>(py) * static_cast<std::size_t>(mw) + static_cast<std::size_t>(px)) * 3;
                            montage[p] = g;
                            montage[p + 1] = g;
                            montage[p + 2] = g;
                        }
                    }
                }
            }
        }

        {
            std::ostringstream name;
            name << "layer_" << li << "_weights.ppm";
            write_ppm_p6(out_dir / name.str(), mw, mh, montage);
        }

        const auto& bvals = B.values();
        float b_min = bvals.front();
        float b_max = bvals.front();
        for (const float v : bvals) {
            b_min = std::min(b_min, v);
            b_max = std::max(b_max, v);
        }
        const float b_span = b_max - b_min;
        const auto norm_b = [&](const float v) -> std::uint8_t {
            if (!(b_span > 0.f)) {
                return 128;
            }
            const float t = (v - b_min) / b_span;
            return static_cast<std::uint8_t>(std::clamp(t * 255.f, 0.f, 255.f));
        };

        constexpr int bias_h = 24;
        const int bw = static_cast<int>(out_f * k_scale);
        std::vector<std::uint8_t> bias_img(static_cast<std::size_t>(bw) * static_cast<std::size_t>(bias_h) * 3, 0);
        for (std::size_t c = 0; c < out_f; ++c) {
            const std::uint8_t g = norm_b(bvals[c]);
            const int x0 = static_cast<int>(c * static_cast<std::size_t>(k_scale));
            for (int y = 0; y < bias_h; ++y) {
                for (int dx = 0; dx < k_scale; ++dx) {
                    const int px = x0 + dx;
                    const std::size_t p = (static_cast<std::size_t>(y) * static_cast<std::size_t>(bw) + static_cast<std::size_t>(px)) * 3;
                    bias_img[p] = g;
                    bias_img[p + 1] = g;
                    bias_img[p + 2] = g;
                }
            }
        }
        {
            std::ostringstream name;
            name << "layer_" << li << "_bias.ppm";
            write_ppm_p6(out_dir / name.str(), bw, bias_h, bias_img);
        }
    }
}

} // namespace tensor_autograd
