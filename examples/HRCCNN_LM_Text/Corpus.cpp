#include "Corpus.h"

#include <cstdio>
#include <fstream>
#include <sstream>

namespace hrccnn_lm_text {

namespace {

void BuildCharTable(Corpus& c)
{
    c.char_to_class.fill(-1);
    for (std::size_t i = 0; i < c.vocab.size(); ++i) {
        const auto u = static_cast<unsigned char>(c.vocab[i]);
        if (u < kVocabCap) c.char_to_class[u] = static_cast<int>(i);
    }
}

std::string MakeFixedVocab()
{
    std::string v;
    v.reserve(kVocabSize);
    v += '\n';                        // 0x0A — newline
    for (char c = 0x20; c <= 0x7E; ++c)  // printable ASCII
        v += c;
    return v;
}

}  // namespace

const std::string& FixedVocab()
{
    static const std::string v = MakeFixedVocab();
    return v;
}

bool LoadCorpus(const std::string& path, Corpus& out)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;

    std::ostringstream buf;
    buf << is.rdbuf();
    out.text = buf.str();
    if (out.text.empty()) return false;

    out.vocab = FixedVocab();
    BuildCharTable(out);

    for (std::size_t i = 0; i < out.text.size(); ++i) {
        if (CharToClass(out, out.text[i]) < 0) {
            std::fprintf(stderr, "error: corpus byte 0x%02X at offset %zu "
                         "is outside the fixed 96-token vocab\n",
                         static_cast<unsigned char>(out.text[i]), i);
            return false;
        }
    }
    return true;
}

void AttachCorpus(const std::string& text, const std::string& vocab, Corpus& out)
{
    out.text  = text;
    out.vocab = vocab;
    BuildCharTable(out);
}

void BipolarBits(char c, float out[kInputBits])
{
    const auto byte = static_cast<unsigned char>(c);
    for (std::size_t b = 0; b < kInputBits; ++b) {
        out[b] = ((byte >> b) & 1u) ? 1.0f : -1.0f;
    }
}

void BipolarEncode(const std::string& s, float* out)
{
    for (std::size_t i = 0; i < s.size(); ++i) {
        BipolarBits(s[i], out + i * kInputBits);
    }
}

}  // namespace hrccnn_lm_text
