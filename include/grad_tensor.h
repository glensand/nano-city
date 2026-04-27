/* Copyright (C) 2026 Gleb Bezborodov - All Rights Reserved
* You may use, distribute and modify this code under the
 * terms of the MIT license.
 *
 * You should have received a copy of the MIT license with
 * this file. If not, please write to: bezborodoff.gleb@gmail.com, or visit : https://github.com/glensand/nano-city
 */

 #pragma once

#include <cassert>
#include <cstddef>
#include <algorithm>

namespace tensor {

class matrix {
    using real_t = float;
public:
    matrix(const std::size_t rows, const std::size_t cols)
        : m_rows(rows)
        , m_cols(cols)
        , m_data(new real_t[rows * cols])
    {
    }
    ~matrix() {
        delete[] m_data;
    }
    real_t& operator()(const std::size_t row, const std::size_t col) {
        return m_data[row * m_cols + col];
    }
    const real_t& operator()(const std::size_t row, const std::size_t col) const {
        return m_data[row * m_cols + col];
    }
    std::size_t rows() const { return m_rows; }
    std::size_t cols() const { return m_cols; }
    std::size_t size() const { return m_rows * m_cols; }
    real_t* data() { return m_data; }
    const real_t* data() const { return m_data; }

    matrix& operator=(const matrix& other) {
        if (this != &other) {
            real_t* new_data = new real_t[other.size()];
            std::copy(other.m_data, other.m_data + other.size(), new_data);
            delete[] m_data;
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            m_data = new_data;
        }
        return *this;
    }

    matrix& operator=(matrix&& other) noexcept {
        if (this != &other) {
            delete[] m_data;
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            m_data = other.m_data;
            other.m_rows = 0;
            other.m_cols = 0;
            other.m_data = nullptr;
        }
        return *this;
    }

    matrix(const matrix& other)
        : m_rows(other.m_rows)
        , m_cols(other.m_cols)
        , m_data(new real_t[other.size()])
    {
        std::copy(other.m_data, other.m_data + other.size(), m_data);
    }

    matrix(matrix&& other) noexcept
        : m_rows(other.m_rows)
        , m_cols(other.m_cols)
        , m_data(other.m_data)
    {
        other.m_data = nullptr;
    }
    
    matrix operator+(const matrix& other) const {
        assert(m_rows == other.m_rows && m_cols == other.m_cols);
        matrix result(m_rows, m_cols);
        for (std::size_t i = 0; i < size(); ++i) {
            result.m_data[i] = m_data[i] + other.m_data[i];
        }
        return result;
    }

    matrix operator*(const matrix& other) const {
        assert(m_cols == other.m_rows);
        matrix result(m_rows, other.m_cols);
        std::fill(result.m_data, result.m_data + result.size(), real_t{ 0 });
        for (std::size_t row = 0; row < m_rows; ++row) {
            for (std::size_t k = 0; k < m_cols; ++k) {
                const real_t lhs = (*this)(row, k);
                for (std::size_t col = 0; col < other.m_cols; ++col) {
                    result(row, col) += lhs * other(k, col);
                }
            }
        }
        return result;
    }

    matrix operator-(const matrix& other) const {
        assert(m_rows == other.m_rows && m_cols == other.m_cols);
        matrix result(m_rows, m_cols);
        for (std::size_t i = 0; i < size(); ++i) {
            result.m_data[i] = m_data[i] - other.m_data[i];
        }
        return result;
    }

    matrix operator/(const matrix& other) const {
        assert(m_rows == other.m_rows && m_cols == other.m_cols);
        matrix result(m_rows, m_cols);
        for (std::size_t i = 0; i < size(); ++i) {
            result.m_data[i] = m_data[i] / other.m_data[i];
        }
        return result;
    }
    
private:

    std::size_t m_rows{ 0 };
    std::size_t m_cols{ 0 };
    real_t* m_data{ nullptr };    
};

}