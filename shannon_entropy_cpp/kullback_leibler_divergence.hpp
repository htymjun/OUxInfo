#ifndef KULLBACK_LEIBLER_DIVERGENCE
#define KULLBACK_LEIBLER_DIVERGENCE


double KL_div(const std::vector<std::vector<double>>& X,
              const std::vector<std::vector<double>>& Y,
              int k = 3);

#endif

