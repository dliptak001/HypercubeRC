#include "Verifier.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "NumberFormat.h"

namespace hrccnn_llm_math {

namespace {

enum class Tok { NUM, OP, LP, RP };
struct Token { Tok kind; double num = 0.0; char op = 0; };

int Prec(char op) { return (op == '+' || op == '-') ? 1 : 2; }

// Tokenize an LHS string. Handles unary minus: a '-' whose prior token is
// OP, LP, or start-of-expression is folded into the following number's sign.
bool Tokenize(const std::string& s, std::vector<Token>& out, std::string* reason)
{
    std::size_t i = 0;
    const std::size_t n = s.size();
    auto prior_allows_unary = [&](void) -> bool {
        if (out.empty()) return true;
        Tok k = out.back().kind;
        return k == Tok::OP || k == Tok::LP;
    };

    while (i < n) {
        char c = s[i];
        if (c == ' ') { ++i; continue; }

        if (c == '(') { out.push_back({Tok::LP});  ++i; continue; }
        if (c == ')') { out.push_back({Tok::RP});  ++i; continue; }

        bool is_digit = (c >= '0' && c <= '9');
        bool is_minus = (c == '-');

        if (is_minus && prior_allows_unary()) {
            // Unary minus: must be followed by a digit.
            if (i + 1 >= n || !(s[i + 1] >= '0' && s[i + 1] <= '9')) {
                if (reason) *reason = "unary minus not followed by digit";
                return false;
            }
            ++i;  // consume '-'
            std::size_t start = i;
            while (i < n && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.')) ++i;
            double v = -std::strtod(s.c_str() + start, nullptr);
            out.push_back({Tok::NUM, v, 0});
            continue;
        }

        if (is_digit) {
            std::size_t start = i;
            while (i < n && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.')) ++i;
            double v = std::strtod(s.c_str() + start, nullptr);
            out.push_back({Tok::NUM, v, 0});
            continue;
        }

        if (c == '+' || c == '-' || c == '*' || c == '/') {
            out.push_back({Tok::OP, 0.0, c});
            ++i;
            continue;
        }

        if (reason) *reason = std::string("unexpected char '") + c + "'";
        return false;
    }
    return true;
}

// Shunting-yard: infix tokens -> RPN.
bool ToRPN(const std::vector<Token>& in, std::vector<Token>& out, std::string* reason)
{
    std::vector<Token> ops;
    for (const auto& t : in) {
        if (t.kind == Tok::NUM) { out.push_back(t); continue; }
        if (t.kind == Tok::OP) {
            while (!ops.empty() && ops.back().kind == Tok::OP &&
                   Prec(ops.back().op) >= Prec(t.op)) {
                out.push_back(ops.back());
                ops.pop_back();
            }
            ops.push_back(t);
            continue;
        }
        if (t.kind == Tok::LP) { ops.push_back(t); continue; }
        if (t.kind == Tok::RP) {
            while (!ops.empty() && ops.back().kind != Tok::LP) {
                out.push_back(ops.back());
                ops.pop_back();
            }
            if (ops.empty()) {
                if (reason) *reason = "mismatched ')'";
                return false;
            }
            ops.pop_back();  // discard LP
        }
    }
    while (!ops.empty()) {
        if (ops.back().kind == Tok::LP) {
            if (reason) *reason = "mismatched '('";
            return false;
        }
        out.push_back(ops.back());
        ops.pop_back();
    }
    return true;
}

// Evaluate RPN token stream. Returns false on div-by-zero or malformed RPN.
bool EvalRPN(const std::vector<Token>& rpn, double& result, std::string* reason)
{
    std::vector<double> st;
    for (const auto& t : rpn) {
        if (t.kind == Tok::NUM) { st.push_back(t.num); continue; }
        if (st.size() < 2) {
            if (reason) *reason = "rpn underflow";
            return false;
        }
        double b = st.back(); st.pop_back();
        double a = st.back(); st.pop_back();
        double v;
        switch (t.op) {
            case '+': v = a + b; break;
            case '-': v = a - b; break;
            case '*': v = a * b; break;
            case '/':
                if (IsCanonicalZero(b)) {
                    if (reason) *reason = "division by canonical zero";
                    return false;
                }
                v = a / b;
                break;
            default:
                if (reason) *reason = "unknown op";
                return false;
        }
        if (!std::isfinite(v)) {
            if (reason) *reason = "non-finite intermediate";
            return false;
        }
        st.push_back(v);
    }
    if (st.size() != 1) {
        if (reason) *reason = "rpn left extra values";
        return false;
    }
    result = st.back();
    return true;
}

}  // namespace

bool VerifyLine(const std::string& line, std::string* reason)
{
    // Require trailing '#'.
    if (line.empty() || line.back() != '#') {
        if (reason) *reason = "missing trailing '#'";
        return false;
    }
    std::string body = line.substr(0, line.size() - 1);

    // Split on " = " (exactly; the separator is load-bearing).
    auto eq = body.find(" = ");
    if (eq == std::string::npos) {
        if (reason) *reason = "missing ' = ' separator";
        return false;
    }
    std::string lhs = body.substr(0, eq);
    std::string rhs = body.substr(eq + 3);

    std::vector<Token> toks;
    if (!Tokenize(lhs, toks, reason)) return false;

    std::vector<Token> rpn;
    if (!ToRPN(toks, rpn, reason)) return false;

    double value;
    if (!EvalRPN(rpn, value, reason)) return false;

    std::string expected_rhs = Canonicalize(value);
    if (expected_rhs != rhs) {
        if (reason) {
            *reason = "RHS mismatch: expected '" + expected_rhs +
                      "', got '" + rhs + "'";
        }
        return false;
    }
    return true;
}

}  // namespace hrccnn_llm_math
