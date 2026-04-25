/* Copyright (C) 2026 Gleb Bezborodov - All Rights Reserved
* You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file. If not, please write to: bezborodoff.gleb@gmail.com, or visit : https://github.com/glensand/nano-city
 */

#pragma once

#include "grad.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <vector>

class module {
public:
    virtual ~module() = default;
    [[nodiscard]] virtual std::vector<scalar*> parameters() = 0;

    void zero_grad() {
        for (auto* parameter : parameters()) {
            parameter->set_grad(0.f);
        }
    }

    void step(const float lr) {
        for (auto* parameter : parameters()) {
            parameter->set_val(parameter->value() - lr * parameter->grad());
        }
    }
};

class neuron final : public module {
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

    [[nodiscard]] const scalar& operator()(const std::vector<const scalar*>& x) const {
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

    [[nodiscard]] std::vector<scalar*> parameters() override {
        std::vector<scalar*> result;
        result.reserve(m_weights.size() + 1);
        for (auto& weight : m_weights) {
            result.push_back(weight.get());
        }
        result.push_back(m_bias.get());
        return result;
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

class layer final : public module {
public:
    layer(const std::size_t nin, const std::size_t nout, const bool nonlin = true) {
        m_neurons.reserve(nout);
        for (std::size_t i = 0; i < nout; ++i) {
            m_neurons.emplace_back(nin, nonlin);
        }
    }

    [[nodiscard]] std::vector<const scalar*> operator()(const std::vector<const scalar*>& x) const {
        std::vector<const scalar*> out;
        out.reserve(m_neurons.size());
        for (const auto& n : m_neurons) {
            out.push_back(&n(x));
        }
        return out;
    }

    [[nodiscard]] std::vector<scalar*> parameters() override {
        std::vector<scalar*> result;
        for (auto& n : m_neurons) {
            auto n_params = n.parameters();
            result.insert(result.end(), n_params.begin(), n_params.end());
        }
        return result;
    }

private:
    std::vector<neuron> m_neurons;
};

class mlp final : public module {
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

    [[nodiscard]] std::vector<const scalar*> operator()(const std::vector<scalar>& x) const {
        std::vector<const scalar*> activations;
        activations.reserve(x.size());
        for (const auto& input : x) {
            activations.push_back(&input);
        }
        for (const auto& layer_instance : m_layers) {
            activations = layer_instance(activations);
        }
        return activations;
    }

    [[nodiscard]] const scalar& forward_one(const std::vector<scalar>& x) const {
        const auto out = (*this)(x);
        assert(out.size() == 1);
        return *out.front();
    }

    [[nodiscard]] std::vector<scalar*> parameters() override {
        std::vector<scalar*> result;
        for (auto& layer_instance : m_layers) {
            auto layer_params = layer_instance.parameters();
            result.insert(result.end(), layer_params.begin(), layer_params.end());
        }
        return result;
    }

private:
    std::vector<layer> m_layers;
};
