#include "Generator.h"

#include <cmath>
#include <optional>
#include <string>

#include "NumberFormat.h"

namespace hrccnn_llm_math {

namespace {

// Apply a binary op to two doubles, rejecting /0 (by *canonical* divisor).
// Returns std::nullopt on reject; otherwise the evaluated value.
// Note: the std::isfinite guard is defensive — Release builds use
// -ffast-math / -ffinite-math-only, under which isfinite() is a no-op.
// Our bounded grammar keeps values well below the double overflow
// threshold, so this is latent rather than load-bearing.
std::optional<double> Apply(double a, char op, double b)
{
    if (op == '/' && IsCanonicalZero(b)) return std::nullopt;
    double v = 0.0;
    switch (op) {
        case '+': v = a + b; break;
        case '-': v = a - b; break;
        case '*': v = a * b; break;
        case '/': v = a / b; break;
        default:  return std::nullopt;
    }
    if (!std::isfinite(v)) return std::nullopt;
    return v;
}

// Print "a_str op b_str" with surrounding spaces on the op.
std::string Join(const std::string& a, char op, const std::string& b)
{
    std::string s;
    s.reserve(a.size() + b.size() + 3);
    s.append(a);
    s.push_back(' ');
    s.push_back(op);
    s.push_back(' ');
    s.append(b);
    return s;
}

// Parenthesize a string: "(s)".
std::string Paren(const std::string& s)
{
    std::string r;
    r.reserve(s.size() + 2);
    r.push_back('(');
    r.append(s);
    r.push_back(')');
    return r;
}

// Finalize a line: "<expr> = <rhs>#", with optional |RHS| filter.
std::optional<std::string> Finalize(const std::string& expr,
                                    double value,
                                    bool filter_999)
{
    if (!std::isfinite(value)) return std::nullopt;
    double rounded = std::round(value * 100.0) / 100.0;
    if (filter_999 && std::fabs(rounded) >= 1000.0) return std::nullopt;
    std::string rhs = Canonicalize(value);
    return expr + " = " + rhs + "#";
}

}  // namespace

Generator::Generator(std::uint64_t seed, GeneratorConfig cfg)
    : rng_(seed), cfg_(cfg)
{}

Generator::Atom Generator::SampleAtom()
{
    // Magnitude in [0, operand_max_magnitude] at 2dp resolution.
    std::uniform_int_distribution<int> cents(
        0, static_cast<int>(std::lround(cfg_.operand_max_magnitude * 100.0)));
    std::bernoulli_distribution neg(cfg_.unary_minus_probability);

    double mag = static_cast<double>(cents(rng_)) / 100.0;
    bool   is_neg = neg(rng_);
    double value = is_neg ? -mag : mag;
    // Canonicalize(value) produces canonical printable form; Canonicalize
    // collapses -0 → "0" so the unary minus naturally disappears on zero.
    return Atom{value, Canonicalize(value)};
}

char Generator::SampleOp()
{
    static const char kOps[4] = {'+', '-', '*', '/'};
    std::uniform_int_distribution<int> d(0, 3);
    return kOps[d(rng_)];
}

std::string Generator::TryOne()
{
    std::bernoulli_distribution depth2(cfg_.depth2_probability);

    if (!depth2(rng_)) {
        // Depth-1: a op b
        Atom a = SampleAtom();
        Atom b = SampleAtom();
        char op = SampleOp();
        auto v = Apply(a.value, op, b.value);
        if (!v) return {};
        auto line = Finalize(Join(a.text, op, b.text), *v, cfg_.rhs_filter_999);
        return line.value_or(std::string{});
    }

    // Depth-2: uniform over {left, right, both}-paren
    std::uniform_int_distribution<int> which(0, 2);
    int kind = which(rng_);
    char op1 = SampleOp();
    char op2 = SampleOp();
    Atom a = SampleAtom();
    Atom b = SampleAtom();
    Atom c = SampleAtom();

    if (kind == 0) {
        // (a op1 b) op2 c
        auto inner = Apply(a.value, op1, b.value);
        if (!inner) return {};
        auto outer = Apply(*inner, op2, c.value);
        if (!outer) return {};
        std::string expr = Join(Paren(Join(a.text, op1, b.text)), op2, c.text);
        auto line = Finalize(expr, *outer, cfg_.rhs_filter_999);
        return line.value_or(std::string{});
    }
    if (kind == 1) {
        // a op1 (b op2 c)
        auto inner = Apply(b.value, op2, c.value);
        if (!inner) return {};
        auto outer = Apply(a.value, op1, *inner);
        if (!outer) return {};
        std::string expr = Join(a.text, op1, Paren(Join(b.text, op2, c.text)));
        auto line = Finalize(expr, *outer, cfg_.rhs_filter_999);
        return line.value_or(std::string{});
    }
    // kind == 2: (a op1 b) op2 (c op3 d)
    char op3 = SampleOp();
    Atom d = SampleAtom();
    auto left  = Apply(a.value, op1, b.value);
    if (!left) return {};
    auto right = Apply(c.value, op3, d.value);
    if (!right) return {};
    auto outer = Apply(*left, op2, *right);
    if (!outer) return {};
    std::string expr = Join(Paren(Join(a.text, op1, b.text)),
                            op2,
                            Paren(Join(c.text, op3, d.text)));
    auto line = Finalize(expr, *outer, cfg_.rhs_filter_999);
    return line.value_or(std::string{});
}

std::string Generator::Sample()
{
    while (true) {
        std::string line = TryOne();
        if (!line.empty()) return line;
    }
}

}  // namespace hrccnn_llm_math
