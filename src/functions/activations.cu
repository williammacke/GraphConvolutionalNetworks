#include "functions/activations.h"

scalar_drelu drelu::r = {};
scalar_relu relu::r = {};
drelu relu::dr = {};

oneSub dsoftmax::os = {};

size_t softmax::cap_ones = 0;
size_t softmax::cap_sum = 0;
float* softmax::ones = nullptr;
float* softmax::sum = nullptr;
scalar_exp softmax::sexp = {};
dsoftmax softmax::ds = {};

addE softmax::ae = {};
