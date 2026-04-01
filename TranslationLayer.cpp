#include "TranslationLayer.h"

// Explicit template instantiations (DIM 5-12)
template std::vector<float> TranslationTransform<5>(const float*, size_t);
template std::vector<float> TranslationTransform<6>(const float*, size_t);
template std::vector<float> TranslationTransform<7>(const float*, size_t);
template std::vector<float> TranslationTransform<8>(const float*, size_t);
template std::vector<float> TranslationTransform<9>(const float*, size_t);
template std::vector<float> TranslationTransform<10>(const float*, size_t);
template std::vector<float> TranslationTransform<11>(const float*, size_t);
template std::vector<float> TranslationTransform<12>(const float*, size_t);

template std::vector<float> TranslationTransformSelected<5>(const float*, size_t, size_t, size_t);
template std::vector<float> TranslationTransformSelected<6>(const float*, size_t, size_t, size_t);
template std::vector<float> TranslationTransformSelected<7>(const float*, size_t, size_t, size_t);
template std::vector<float> TranslationTransformSelected<8>(const float*, size_t, size_t, size_t);
template std::vector<float> TranslationTransformSelected<9>(const float*, size_t, size_t, size_t);
template std::vector<float> TranslationTransformSelected<10>(const float*, size_t, size_t, size_t);
template std::vector<float> TranslationTransformSelected<11>(const float*, size_t, size_t, size_t);
template std::vector<float> TranslationTransformSelected<12>(const float*, size_t, size_t, size_t);
