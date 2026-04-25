/* Copyright (C) 2026 Gleb Bezborodov - All Rights Reserved
* You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file. If not, please write to: bezborodoff.gleb@gmail.com, or visit : https://github.com/glensand/nano-city
 */

#pragma once

#include <deque>
#include <vector>
#include <cassert>
#include <cmath>
#include <unordered_set>
#include <optional>
#include <ranges>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <string>

class scalar_graph_viz;

class scalar final {
    using T = float;
public:
    explicit scalar(const T& in_val, std::vector<scalar*> children = {}, const char* label = "Const") {
        m_val = in_val;
        m_children = std::move(children);
        m_label = label;
    }

    ~scalar() {
        for (auto v : m_controlled_constants) {
            delete v;
        }
    }

    const scalar& operator*(const scalar& rhs) const {
        const auto mul_val = rhs.m_val * m_val;
        auto res = alloc_and_return(mul_val, { (scalar*)this, (scalar*)&rhs }, "*");
        res->m_backward = &scalar::mul_backward;
        return *res;
    }

    const scalar& operator*(const float& rhs) {
        auto new_var = new scalar(rhs);
        m_controlled_constants.push_back(new_var);
        return *this * *new_var;
    }

    const scalar& operator+(const scalar& rhs) const {
        const auto add_val = rhs.m_val + m_val;
        auto res = alloc_and_return(add_val, { (scalar*)this, (scalar*)&rhs}, "+");
        res->m_backward = &scalar::add_backward;
        return *res;
    }

    const scalar& operator+(const float& rhs) const {
        auto new_var = new scalar(rhs);
        m_controlled_constants.push_back(new_var);
        return *this + *new_var;
    }
    
    const scalar& pow(const scalar& p) const {
        const auto val = std::pow(m_val, p.m_val);
        auto new_var = alloc_and_return(val, { (scalar*)this, (scalar*)&p }, "^");
        new_var->m_backward = &scalar::pow_backward;
        return *new_var;
    }

    const scalar& pow(const float& p) const {
        auto new_var = new scalar(p);
        m_controlled_constants.push_back(new_var);
        return pow(*new_var);
    }

    const scalar& relu() const {
        const T val = m_val < 0 ? 0.f : m_val;
        auto res = alloc_and_return(val, { (scalar*)this }, "relu");
        res->m_backward = &scalar::relu_backward;
        return *res;
    }

    void set_grad(const T& grad) { m_grad = grad; }
    void set_val(const T& val) { m_val = val; }
    [[nodiscard]] T value() const { return m_val; }
    [[nodiscard]] T grad() const { return m_grad; }
    
    void backward() {
        if (!m_topo.has_value()){
            std::unordered_set<scalar*> visited;
            std::deque<scalar*> pending_visit;
            std::deque<scalar*> reverse_topo;
            pending_visit.push_back(this);
            visited.insert(this);
            while (!pending_visit.empty()) {
                auto* node = pending_visit.back();
                pending_visit.pop_back();
                reverse_topo.push_back(node);

                for (auto* child : node->m_children) {
                    if (visited.insert(child).second) {
                        pending_visit.push_back(child);
                    }
                }
            }

            m_topo = std::vector<scalar*>{};
            while (!reverse_topo.empty()) {
                m_topo->push_back(reverse_topo.back());
                reverse_topo.pop_back();
            }
        }
        for (auto* node : std::views::reverse(*m_topo)) {
            if (node->m_backward != nullptr) {
                (node->*node->m_backward)();
            }
        }
    }

    void reset_grad() {
        if (m_topo.has_value()) {
            for (auto v : m_topo.value()) {
                v->m_grad = {};
            }
        }
    }

    void reset_topo(){
        m_topo = std::nullopt;
    }

private:
    friend class scalar_graph_viz;

    scalar* alloc_and_return(const T& in_val, std::vector<scalar*> children = {}, const char* label = "Const") const {
        auto new_var = new scalar(in_val, std::move(children), label);
        m_controlled_constants.push_back(new_var);
        return new_var;
    }

    void mul_backward() {
        assert(m_children.size() == 2);
        m_children[0]->m_grad += m_children[1]->m_val * m_grad;
        m_children[1]->m_grad += m_children[0]->m_val * m_grad;
    }

    void add_backward() {
        assert(m_children.size() == 2);
        m_children[0]->m_grad += m_grad;
        m_children[1]->m_grad += m_grad;
    }
    
    void pow_backward() {
        assert(m_children.size() == 2);
        const auto p = m_children[1]->m_val;
        m_children[0]->m_grad += (p * std::pow(m_children[0]->m_val, (p - 1))) * m_grad;
    }

    void relu_backward() {
        assert(m_children.size() == 1);
        m_children[0]->m_grad += (m_val > 0) * m_grad;
    }
    
    std::optional<std::vector<scalar*>> m_topo;
    mutable std::vector<scalar*> m_controlled_constants;
    std::vector<scalar*> m_children;
    const char* m_label{ nullptr };

    T m_val{ 0 };
    T m_grad{ 0 };

    void (scalar::*m_backward)(){}; 
};

class scalar_graph_viz final {
public:
    [[nodiscard]] static std::string to_dot(const scalar& root) {
        std::unordered_set<const scalar*> visited;
        std::deque<const scalar*> stack;
        std::ostringstream ss;

        ss << "digraph ComputationalGraph {\n";
        ss << "  rankdir=LR;\n";
        ss << "  node [shape=record, fontname=\"Helvetica\"];\n";

        stack.push_back(&root);
        visited.insert(&root);

        while (!stack.empty()) {
            const auto* node = stack.back();
            stack.pop_back();

            const auto node_id = id_of(node);
            ss << "  " << node_id << " [label=\"{"
               << sanitize(node->m_label)
               << "|value=" << std::setprecision(6) << node->m_val
               << "|grad=" << std::setprecision(6) << node->m_grad
               << "}\"];\n";

            if (!node->m_children.empty()) {
                const auto op_id = node_id + "_op";
                ss << "  " << op_id << " [shape=oval, label=\""
                   << sanitize(node->m_label) << "\"];\n";
                ss << "  " << op_id << " -> " << node_id << ";\n";
                for (const auto* child : node->m_children) {
                    ss << "  " << id_of(child) << " -> " << op_id << ";\n";
                    if (visited.insert(child).second) {
                        stack.push_back(child);
                    }
                }
            }
        }

        ss << "}\n";
        return ss.str();
    }

    static void dump_dot(const scalar& root, const std::string& path) {
        std::ofstream out(path);
        out << to_dot(root);
    }

private:
    static std::string id_of(const scalar* node) {
        std::ostringstream ss;
        ss << "n" << std::hex << reinterpret_cast<std::uintptr_t>(node);
        return ss.str();
    }

    static std::string sanitize(const char* label) {
        if (label == nullptr) {
            return "null";
        }

        std::string out;
        for (const char c : std::string(label)) {
            if (c == '\"') {
                out += "\\\"";
            } else if (c == '\\') {
                out += "\\\\";
            } else {
                out += c;
            }
        }
        return out;
    }
};
