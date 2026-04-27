#pragma once

#include <cassert>
#include <cstddef>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

namespace scalar_static {

class graph final {
public:
    using real_t = float;

    enum class op_type {
        leaf,
        add,
        mul,
        relu
    };

    struct node final {
        op_type op{ op_type::leaf };
        node* lhs{ nullptr };
        node* rhs{ nullptr };
        real_t value{ 0.f };
        real_t grad{ 0.f };
        bool trainable{ false };
    };

    [[nodiscard]] node* leaf(const real_t value, const bool trainable = false) {
        return make_node(op_type::leaf, nullptr, nullptr, value, trainable);
    }

    [[nodiscard]] node* add(node* lhs, node* rhs) {
        return make_node(op_type::add, lhs, rhs, 0.f, false);
    }

    [[nodiscard]] node* mul(node* lhs, node* rhs) {
        return make_node(op_type::mul, lhs, rhs, 0.f, false);
    }

    [[nodiscard]] node* relu(node* in) {
        return make_node(op_type::relu, in, nullptr, 0.f, false);
    }

    [[nodiscard]] std::vector<node*> build_topo(const std::vector<node*>& outputs) const {
        std::vector<node*> topo;
        topo.reserve(m_nodes.size());
        std::unordered_set<const node*> visited;
        visited.reserve(m_nodes.size());
        for (auto* output : outputs) {
            dfs(output, visited, topo);
        }
        return topo;
    }

    void forward(const std::vector<node*>& topo) const {
        for (auto* n : topo) {
            switch (n->op) {
            case op_type::leaf:
                break;
            case op_type::add:
                n->value = n->lhs->value + n->rhs->value;
                break;
            case op_type::mul:
                n->value = n->lhs->value * n->rhs->value;
                break;
            case op_type::relu:
                n->value = n->lhs->value < 0.f ? 0.f : n->lhs->value;
                break;
            default:
                assert(false && "Unknown op_type");
                break;
            }
        }
    }

    void backward(const std::vector<node*>& topo, node* loss, const real_t dloss = 1.f) const {
        for (auto* n : topo) {
            n->grad = 0.f;
        }
        loss->grad = dloss;

        for (std::size_t i = topo.size(); i > 0; --i) {
            auto* n = topo[i - 1];
            switch (n->op) {
            case op_type::leaf:
                break;
            case op_type::add:
                n->lhs->grad += n->grad;
                n->rhs->grad += n->grad;
                break;
            case op_type::mul:
                n->lhs->grad += n->rhs->value * n->grad;
                n->rhs->grad += n->lhs->value * n->grad;
                break;
            case op_type::relu:
                n->lhs->grad += (n->lhs->value > 0.f ? 1.f : 0.f) * n->grad;
                break;
            default:
                assert(false && "Unknown op_type");
                break;
            }
        }
    }

private:
    [[nodiscard]] node* make_node(
        const op_type op,
        node* lhs,
        node* rhs,
        const real_t value,
        const bool trainable)
    {
        auto n = std::make_unique<node>();
        n->op = op;
        n->lhs = lhs;
        n->rhs = rhs;
        n->value = value;
        n->trainable = trainable;
        m_nodes.push_back(std::move(n));
        return m_nodes.back().get();
    }

    static void dfs(
        node* n,
        std::unordered_set<const node*>& visited,
        std::vector<node*>& topo)
    {
        if (n == nullptr || visited.contains(n)) {
            return;
        }
        visited.insert(n);
        dfs(n->lhs, visited, topo);
        dfs(n->rhs, visited, topo);
        topo.push_back(n);
    }

    std::vector<std::unique_ptr<node>> m_nodes;
};

} // namespace scalar_static
