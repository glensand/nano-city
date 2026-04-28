#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

namespace tensor_autograd {

class tensor final {
public:
    using real_t = float;

    tensor() = default;

    tensor(
        const std::size_t rows,
        const std::size_t cols,
        const real_t init = 0.f,
        const bool requires_grad = false)
        : m_node(std::make_shared<node>(rows, cols, init, requires_grad))
    {
    }

    static tensor from_data(
        const std::size_t rows,
        const std::size_t cols,
        const std::vector<real_t>& values,
        const bool requires_grad = false)
    {
        assert(values.size() == rows * cols);
        tensor t(rows, cols, 0.f, requires_grad);
        t.m_node->data = values;
        return t;
    }

    static tensor random_uniform(
        const std::size_t rows,
        const std::size_t cols,
        const real_t limit,
        const bool requires_grad = true)
    {
        static std::mt19937 rng(1337);
        std::uniform_real_distribution<real_t> dist(-limit, limit);
        tensor t(rows, cols, 0.f, requires_grad);
        for (auto& v : t.m_node->data) {
            v = dist(rng);
        }
        return t;
    }

    [[nodiscard]] std::size_t rows() const { return m_node->rows; }
    [[nodiscard]] std::size_t cols() const { return m_node->cols; }
    [[nodiscard]] std::size_t size() const { return m_node->data.size(); }
    [[nodiscard]] bool requires_grad() const { return m_node->requires_grad; }

    [[nodiscard]] const std::vector<real_t>& values() const { return m_node->data; }
    [[nodiscard]] const std::vector<real_t>& grads() const { return m_node->grad; }

    real_t& operator()(const std::size_t row, const std::size_t col) {
        return m_node->data[row * cols() + col];
    }

    const real_t& operator()(const std::size_t row, const std::size_t col) const {
        return m_node->data[row * cols() + col];
    }

    tensor operator+(const tensor& other) const {
        assert(rows() == other.rows() && cols() == other.cols());
        tensor out(rows(), cols(), 0.f, requires_grad() || other.requires_grad());
        for (std::size_t i = 0; i < size(); ++i) {
            out.m_node->data[i] = m_node->data[i] + other.m_node->data[i];
        }
        out.m_node->op = op_type::add;
        out.m_node->lhs = m_node;
        out.m_node->rhs = other.m_node;
        return out;
    }

    tensor operator-(const tensor& other) const {
        assert(rows() == other.rows() && cols() == other.cols());
        tensor out(rows(), cols(), 0.f, requires_grad() || other.requires_grad());
        for (std::size_t i = 0; i < size(); ++i) {
            out.m_node->data[i] = m_node->data[i] - other.m_node->data[i];
        }
        out.m_node->op = op_type::sub;
        out.m_node->lhs = m_node;
        out.m_node->rhs = other.m_node;
        return out;
    }

    tensor matmul(const tensor& other) const {
        assert(cols() == other.rows());
        tensor out(rows(), other.cols(), 0.f, requires_grad() || other.requires_grad());
        for (std::size_t r = 0; r < rows(); ++r) {
            for (std::size_t k = 0; k < cols(); ++k) {
                const real_t lhs = (*this)(r, k);
                for (std::size_t c = 0; c < other.cols(); ++c) {
                    out(r, c) += lhs * other(k, c);
                }
            }
        }
        out.m_node->op = op_type::matmul;
        out.m_node->lhs = m_node;
        out.m_node->rhs = other.m_node;
        return out;
    }

    tensor add_rowwise(const tensor& row_vector) const {
        assert(row_vector.rows() == 1 && row_vector.cols() == cols());
        tensor out(rows(), cols(), 0.f, requires_grad() || row_vector.requires_grad());
        for (std::size_t r = 0; r < rows(); ++r) {
            for (std::size_t c = 0; c < cols(); ++c) {
                out(r, c) = (*this)(r, c) + row_vector(0, c);
            }
        }
        out.m_node->op = op_type::add_rowwise;
        out.m_node->lhs = m_node;
        out.m_node->rhs = row_vector.m_node;
        return out;
    }

    tensor relu() const {
        tensor out(rows(), cols(), 0.f, requires_grad());
        for (std::size_t i = 0; i < size(); ++i) {
            out.m_node->data[i] = std::max<real_t>(0.f, m_node->data[i]);
        }
        out.m_node->op = op_type::relu;
        out.m_node->lhs = m_node;
        return out;
    }

    tensor square() const {
        tensor out(rows(), cols(), 0.f, requires_grad());
        for (std::size_t i = 0; i < size(); ++i) {
            out.m_node->data[i] = m_node->data[i] * m_node->data[i];
        }
        out.m_node->op = op_type::square;
        out.m_node->lhs = m_node;
        return out;
    }

    tensor mean() const {
        assert(size() > 0);
        tensor out(1, 1, 0.f, requires_grad());
        for (const auto v : m_node->data) {
            out.m_node->data[0] += v;
        }
        out.m_node->data[0] /= static_cast<real_t>(size());
        out.m_node->op = op_type::mean;
        out.m_node->lhs = m_node;
        return out;
    }

    void zero_grad() {
        std::fill(m_node->grad.begin(), m_node->grad.end(), 0.f);
    }

    void step(const real_t lr) {
        assert(requires_grad());
        for (std::size_t i = 0; i < size(); ++i) {
            m_node->data[i] -= lr * m_node->grad[i];
        }
    }

    void backward() const {
        assert(rows() == 1 && cols() == 1);
        std::vector<node*> topo;
        topo.reserve(256);
        std::unordered_set<const node*> visited;
        visited.reserve(256);
        build_topo(m_node.get(), visited, topo);

        for (auto* n : topo) {
            std::fill(n->grad.begin(), n->grad.end(), 0.f);
        }
        m_node->grad[0] = 1.f;

        for (std::size_t i = topo.size(); i > 0; --i) {
            apply_backward(*topo[i - 1]);
        }
    }

private:
    enum class op_type {
        leaf,
        add,
        sub,
        matmul,
        add_rowwise,
        relu,
        square,
        mean
    };

    struct node final {
        node(
            const std::size_t in_rows,
            const std::size_t in_cols,
            const real_t init,
            const bool in_requires_grad)
            : rows(in_rows)
            , cols(in_cols)
            , data(in_rows * in_cols, init)
            , grad(in_rows * in_cols, 0.f)
            , requires_grad(in_requires_grad)
        {
        }

        std::size_t rows{ 0 };
        std::size_t cols{ 0 };
        std::vector<real_t> data;
        std::vector<real_t> grad;
        bool requires_grad{ false };
        op_type op{ op_type::leaf };
        std::shared_ptr<node> lhs;
        std::shared_ptr<node> rhs;
    };

    static void build_topo(
        node* n,
        std::unordered_set<const node*>& visited,
        std::vector<node*>& topo)
    {
        if (n == nullptr || visited.contains(n)) {
            return;
        }
        visited.insert(n);
        build_topo(n->lhs.get(), visited, topo);
        build_topo(n->rhs.get(), visited, topo);
        topo.push_back(n);
    }

    static void apply_backward(node& n) {
        switch (n.op) {
        case op_type::leaf:
            break;
        case op_type::add: {
            if (n.lhs->requires_grad) {
                for (std::size_t i = 0; i < n.grad.size(); ++i) {
                    n.lhs->grad[i] += n.grad[i];
                }
            }
            if (n.rhs->requires_grad) {
                for (std::size_t i = 0; i < n.grad.size(); ++i) {
                    n.rhs->grad[i] += n.grad[i];
                }
            }
            break;
        }
        case op_type::sub: {
            if (n.lhs->requires_grad) {
                for (std::size_t i = 0; i < n.grad.size(); ++i) {
                    n.lhs->grad[i] += n.grad[i];
                }
            }
            if (n.rhs->requires_grad) {
                for (std::size_t i = 0; i < n.grad.size(); ++i) {
                    n.rhs->grad[i] -= n.grad[i];
                }
            }
            break;
        }
        case op_type::matmul: {
            const auto m = n.rows;
            const auto p = n.cols;
            const auto kdim = n.lhs->cols;
            if (n.lhs->requires_grad) {
                for (std::size_t r = 0; r < m; ++r) {
                    for (std::size_t k = 0; k < kdim; ++k) {
                        real_t acc = 0.f;
                        for (std::size_t c = 0; c < p; ++c) {
                            acc += n.grad[r * p + c] * n.rhs->data[k * p + c];
                        }
                        n.lhs->grad[r * kdim + k] += acc;
                    }
                }
            }
            if (n.rhs->requires_grad) {
                for (std::size_t k = 0; k < kdim; ++k) {
                    for (std::size_t c = 0; c < p; ++c) {
                        real_t acc = 0.f;
                        for (std::size_t r = 0; r < m; ++r) {
                            acc += n.lhs->data[r * kdim + k] * n.grad[r * p + c];
                        }
                        n.rhs->grad[k * p + c] += acc;
                    }
                }
            }
            break;
        }
        case op_type::add_rowwise: {
            if (n.lhs->requires_grad) {
                for (std::size_t i = 0; i < n.grad.size(); ++i) {
                    n.lhs->grad[i] += n.grad[i];
                }
            }
            if (n.rhs->requires_grad) {
                for (std::size_t c = 0; c < n.cols; ++c) {
                    real_t acc = 0.f;
                    for (std::size_t r = 0; r < n.rows; ++r) {
                        acc += n.grad[r * n.cols + c];
                    }
                    n.rhs->grad[c] += acc;
                }
            }
            break;
        }
        case op_type::relu: {
            if (n.lhs->requires_grad) {
                for (std::size_t i = 0; i < n.grad.size(); ++i) {
                    n.lhs->grad[i] += (n.lhs->data[i] > 0.f ? 1.f : 0.f) * n.grad[i];
                }
            }
            break;
        }
        case op_type::square: {
            if (n.lhs->requires_grad) {
                for (std::size_t i = 0; i < n.grad.size(); ++i) {
                    n.lhs->grad[i] += 2.f * n.lhs->data[i] * n.grad[i];
                }
            }
            break;
        }
        case op_type::mean: {
            if (n.lhs->requires_grad) {
                const real_t scale = 1.f / static_cast<real_t>(n.lhs->data.size());
                for (std::size_t i = 0; i < n.lhs->data.size(); ++i) {
                    n.lhs->grad[i] += scale * n.grad[0];
                }
            }
            break;
        }
        default:
            assert(false && "Unknown op type");
            break;
        }
    }

    std::shared_ptr<node> m_node;
};

} // namespace tensor_autograd
