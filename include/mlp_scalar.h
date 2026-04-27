/* Copyright (C) 2026 Gleb Bezborodov - All Rights Reserved
* You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file. If not, please write to: bezborodoff.gleb@gmail.com, or visit : https://github.com/glensand/nano-city
 */

#pragma once

#include "grad_scalar.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <vector>

namespace scalar {

class neuron final {
public:
    neuron(const std::size_t nin, const bool nonlin = true)
        : m_nonlin(nonlin)
    {
        m_weights.reserve(nin);
        const float limit = 1.f / std::sqrt(static_cast<float>(nin));
        for (std::size_t i = 0; i < nin; ++i) {
            m_weights.push_back(std::make_unique<scalar>(rand_uniform(limit), std::vector<scalar*>{}, "w"));
        }
        m_bias = std::make_unique<scalar>(0.f, std::vector<scalar*>{}, "b");
    }

    [[nodiscard]] const scalar& forward_scalar(const std::vector<const scalar*>& x) const {
        assert(x.size() == m_weights.size());

        const scalar* out = m_bias.get();
        for (std::size_t i = 0; i < x.size(); ++i) {
            out = &((*out) + ((*m_weights[i]) * (*x[i])));
        }
        if (m_nonlin) {
            out = &(out->relu());
        }
        return *out;
    }

    [[nodiscard]] float forward(const std::vector<float>& x) const {
        assert(x.size() == m_weights.size());
        float out = m_bias->value();
        for (std::size_t i = 0; i < x.size(); ++i) {
            out += m_weights[i]->value() * x[i];
        }
        if (m_nonlin) {
            out = out < 0.f ? 0.f : out;
        }
        return out;
    }

    void zero_grad() {
        for (auto& weight : m_weights) {
            weight->set_grad(0.f);
        }
        m_bias->set_grad(0.f);
    }

    void step(const float lr) {
        for (auto& weight : m_weights) {
            weight->set_val(weight->value() - lr * weight->grad());
        }
        m_bias->set_val(m_bias->value() - lr * m_bias->grad());
    }

private:
    static float rand_uniform(const float limit) {
        static std::mt19937 rng(1337);
        std::uniform_real_distribution<float> dist(-limit, limit);
        return dist(rng);
    }

    std::vector<std::unique_ptr<scalar>> m_weights;
    std::unique_ptr<scalar> m_bias;
    bool m_nonlin{ true };
};

class layer final {
public:
    layer(const std::size_t nin, const std::size_t nout, const bool nonlin = true) {
        m_neurons.reserve(nout);
        for (std::size_t i = 0; i < nout; ++i) {
            m_neurons.emplace_back(nin, nonlin);
        }
    }

    [[nodiscard]] std::vector<const scalar*> forward_scalar(const std::vector<const scalar*>& x) const {
        std::vector<const scalar*> out;
        out.reserve(m_neurons.size());
        for (const auto& n : m_neurons) {
            out.push_back(&n.forward_scalar(x));
        }
        return out;
    }

    [[nodiscard]] std::vector<float> forward(const std::vector<float>& x) const {
        std::vector<float> out;
        out.reserve(m_neurons.size());
        for (const auto& n : m_neurons) {
            out.push_back(n.forward(x));
        }
        return out;
    }

    void zero_grad() {
        for (auto& n : m_neurons) {
            n.zero_grad();
        }
    }

    void step(const float lr) {
        for (auto& n : m_neurons) {
            n.step(lr);
        }
    }

private:
    std::vector<neuron> m_neurons;
};

class mlp final {
public:
    mlp(const std::size_t nin, const std::vector<std::size_t>& nouts) {
        std::size_t in = nin;
        m_layers.reserve(nouts.size());
        for (std::size_t i = 0; i < nouts.size(); ++i) {
            const bool nonlin = i + 1 != nouts.size();
            m_layers.emplace_back(in, nouts[i], nonlin);
            in = nouts[i];
        }
    }

    [[nodiscard]] std::vector<const scalar*> forward_scalar(const std::vector<scalar>& x) const {
        std::vector<const scalar*> activations;
        activations.reserve(x.size());
        for (const auto& input : x) {
            activations.push_back(&input);
        }
        for (const auto& layer_instance : m_layers) {
            activations = layer_instance.forward_scalar(activations);
        }
        return activations;
    }

    [[nodiscard]] const scalar& error_head(
        const std::vector<scalar>& x,
        const std::vector<float>& target) const
    {
        const auto logits = forward_scalar(x);
        assert(logits.size() == target.size());
        assert(!logits.empty());

        const scalar* total = &(((*logits[0]) + (-target[0])).pow(2.f));
        for (std::size_t i = 1; i < logits.size(); ++i) {
            total = &((*total) + (((*logits[i]) + (-target[i])).pow(2.f)));
        }
        return *total;
    }

    void backward_from_loss(const scalar& loss, const float dloss = 1.f) const {
        auto& loss_node = const_cast<scalar&>(loss);
        loss_node.reset_grad();
        loss_node.set_grad(dloss);
        loss_node.backward();
    }

    [[nodiscard]] std::vector<float> forward(const std::vector<float>& x) const {
        std::vector<float> activations = x;
        for (const auto& layer_instance : m_layers) {
            activations = layer_instance.forward(activations);
        }
        return activations;
    }

    [[nodiscard]] float forward_one(const std::vector<float>& x) const {
        const auto out = forward(x);
        assert(out.size() == 1);
        return out.front();
    }

    void zero_grad() {
        for (auto& layer_instance : m_layers) {
            layer_instance.zero_grad();
        }
    }

    void step(const float lr) {
        for (auto& layer_instance : m_layers) {
            layer_instance.step(lr);
        }
    }

private:
    std::vector<layer> m_layers;
};

}
