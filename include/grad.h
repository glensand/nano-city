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

// class base_var {
// public:
//     virtual ~base_var() = default;
// };

class var final {
    using T = float;
public:
    explicit var(const T& in_val, std::vector<var*> children = {}, const char* label = "constant") {
        m_val = in_val;
        m_children = std::move(children);
        m_label = label;
    }

    ~var() {
        for (auto v : controlled_constants) {
            delete v;
        }
    }

    var operator*(const var& rhs) {
        const auto mul_val = rhs.m_val * m_val;
        var res(mul_val, { this, (var*)&rhs}, "*");
        res.m_backward = &var::mul_backward;
        return res;
    }

    var operator*(const float& rhs) {
        auto new_var = new var(rhs);
        controlled_constants.push_back(new_var);
        return *this * *new_var;
    }

    var operator+(const var& rhs){
        const auto add_val = rhs.m_val * m_val;
        var res(add_val, { this, (var*)&rhs}, "+");
        res.m_backward = &var::add_backward;
        return res;
    }

    var operator+(const float& rhs){
        auto new_var = new var(rhs);
        controlled_constants.push_back(new_var);
        return *this + *new_var;
    }
    
    var pow(const var& p) {
        const auto val = std::pow(m_val, p.m_val);
        var res(val, { this, (var*)&p }, "^");
        res.m_backward = &var::pow_backward;
        return res;
    }

    var pow(const float& p) {
        auto new_var = new var(p);
        controlled_constants.push_back(new_var);
        return pow(*new_var);
    }

    var relu(){
        const T val = m_val < 0 ? 0.f : m_val;
        var res(val, { this }, "relu");
        res.m_backward = &var::relu_backward;
        return res;
    }

    void set_grad(const T& grad) { m_grad = grad; }
    void set_val(const T& val) { m_val = val; }
    
    void backward() {
        std::unordered_set<var*> visited;
        std::deque<var*> pending_visit;
        std::vector<var*> topo;
        pending_visit.push_back(this);
        while (!pending_visit.empty()) {

        }
    }

private:

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
    
    std::vector<var*> controlled_constants;
    std::vector<var*> m_children;
    const char* m_label{ nullptr };

    T m_val{ 0 };
    T m_grad{ 0 };

    void (var::*m_backward)(); 
};