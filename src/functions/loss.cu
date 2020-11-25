#include "functions/loss.h"

oneSum dcross_entropy_with_logits::os = {};

float scalar_log::epsilon = 0.0001f;

size_t cross_entropy_with_logits::cap_ones = 0;
float* cross_entropy_with_logits::ones = nullptr;
size_t cross_entropy_with_logits::cap_tmp = 0;
float* cross_entropy_with_logits::tmp = nullptr;
size_t cross_entropy_with_logits::cap_tmp2 = 0;
float* cross_entropy_with_logits::tmp2 = nullptr;
scalar_log cross_entropy_with_logits::sl = {};
oneSum cross_entropy_with_logits::os = {};
dcross_entropy_with_logits cross_entropy_with_logits::dc = {};
