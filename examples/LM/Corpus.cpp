#include "Corpus.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace lm {

Corpus Corpus::LoadFile(const std::string& path, const Vocabulary& vocab)
{
    std::ifstream is(path, std::ios::binary);
    if (!is)
        throw std::runtime_error("could not open corpus file: " + path);

    std::ostringstream buf;
    buf << is.rdbuf();
    std::string text = buf.str();

    if (text.empty())
        throw std::runtime_error("corpus file is empty: " + path);

    for (std::size_t i = 0; i < text.size(); ++i) {
        if (!vocab.Contains(text[i])) {
            char msg[128];
            std::snprintf(msg, sizeof(msg),
                          "corpus byte 0x%02X at offset %zu is outside vocabulary",
                          static_cast<unsigned char>(text[i]), i);
            throw std::runtime_error(msg);
        }
    }

    return Corpus(std::move(text));
}

Corpus::Corpus(std::string text)
    : text_(std::move(text))
{
}

}  // namespace lm
