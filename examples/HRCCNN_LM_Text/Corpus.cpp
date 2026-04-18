#include "Corpus.h"

#include <algorithm>
#include <fstream>
#include <set>
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

}  // namespace

bool LoadCorpus(const std::string& path, Corpus& out)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;

    std::ostringstream buf;
    buf << is.rdbuf();
    out.text = buf.str();
    if (out.text.empty()) return false;

    std::set<char> unique(out.text.begin(), out.text.end());
    out.vocab.assign(unique.begin(), unique.end());
    BuildCharTable(out);
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
