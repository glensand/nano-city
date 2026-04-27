#pragma once

#include "grad_scalar_static.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace scalar_static {

class mlp final {
public:
    mlp(const std::size_t nin, const std::vector<std::size_t>& nouts) {
        assert(!nouts.empty());
        m_inputs.reserve(nin);
        for (std::size_t i = 0; i < nin; ++i) {
            m_inputs.push_back(m_graph.leaf(0.f, false));
        }

        std::vector<graph::node*> activations = m_inputs;
        std::size_t in = nin;
        for (std::size_t layer_idx = 0; layer_idx < nouts.size(); ++layer_idx) {
            const std::size_t out = nouts[layer_idx];
            const bool nonlin = layer_idx + 1 != nouts.size();
            std::vector<graph::node*> next;
            next.reserve(out);

            for (std::size_t j = 0; j < out; ++j) {
                auto* acc = m_graph.leaf(0.f, true);
                m_parameters.push_back(acc); // bias
                for (std::size_t i = 0; i < in; ++i) {
                    auto* w = m_graph.leaf(rand_uniform(1.f / std::sqrt(static_cast<float>(in))), true);
                    m_parameters.push_back(w);
                    acc = m_graph.add(acc, m_graph.mul(w, activations[i]));
                }
                if (nonlin) {
                    acc = m_graph.relu(acc);
                }
                next.push_back(acc);
            }

            activations = std::move(next);
            in = out;
        }

        m_logits = activations;
        m_targets.reserve(m_logits.size());
        for (std::size_t i = 0; i < m_logits.size(); ++i) {
            m_targets.push_back(m_graph.leaf(0.f, false));
        }

        auto* neg_one = m_graph.leaf(-1.f, false);
        for (std::size_t i = 0; i < m_logits.size(); ++i) {
            auto* diff = m_graph.add(m_logits[i], m_graph.mul(neg_one, m_targets[i]));
            auto* sq = m_graph.mul(diff, diff);
            m_loss = (m_loss == nullptr) ? sq : m_graph.add(m_loss, sq);
        }

        m_topo_logits = m_graph.build_topo(m_logits);
        m_topo_loss = m_graph.build_topo({ m_loss });
        m_logits_cache.resize(m_logits.size(), 0.f);
    }

    [[nodiscard]] const std::vector<float>& forward(const std::vector<float>& x) {
        assert(x.size() == m_inputs.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            m_inputs[i]->value = x[i];
        }
        m_graph.forward(m_topo_logits);
        for (std::size_t i = 0; i < m_logits.size(); ++i) {
            m_logits_cache[i] = m_logits[i]->value;
        }
        return m_logits_cache;
    }

    [[nodiscard]] float forward_loss(const std::vector<float>& x, const std::uint8_t label) {
        set_target_one_hot(label);
        (void)forward(x);
        m_graph.forward(m_topo_loss);
        return m_loss->value;
    }

    [[nodiscard]] const graph::node& error_head() const {
        return *m_loss;
    }

    void set_loss_grad(const float dloss) {
        m_dloss = dloss;
    }

    void backward() {
        m_graph.backward(m_topo_loss, m_loss, m_dloss);
    }

    void step(const float lr) {
        for (auto* parameter : m_parameters) {
            parameter->value -= lr * parameter->grad;
        }
    }

private:
    static float rand_uniform(const float limit) {
        static std::mt19937 rng(1337);
        std::uniform_real_distribution<float> dist(-limit, limit);
        return dist(rng);
    }

    void set_target_one_hot(const std::uint8_t label) {
        assert(static_cast<std::size_t>(label) < m_targets.size());
        for (std::size_t i = 0; i < m_targets.size(); ++i) {
            m_targets[i]->value = (i == static_cast<std::size_t>(label)) ? 1.f : 0.f;
        }
    }

    graph m_graph;
    std::vector<graph::node*> m_inputs;
    std::vector<graph::node*> m_targets;
    std::vector<graph::node*> m_parameters;
    std::vector<graph::node*> m_logits;
    graph::node* m_loss{ nullptr };

    std::vector<graph::node*> m_topo_logits;
    std::vector<graph::node*> m_topo_loss;
    std::vector<float> m_logits_cache;
    float m_dloss{ 1.f };
};

} // namespace scalar_static
