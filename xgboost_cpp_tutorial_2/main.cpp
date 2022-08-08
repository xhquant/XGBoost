#include <iostream>
#include <string>
#include <fstream>

#include "xgboost/c_api.h"
#include "data_iterator.hpp"

int main(int argc, char **argv)
{
    const int N_BATCH = 32;
    const int BATCH_LENGTH = 512;

    data_iterator iter{};
    data_iterator_init(&iter, BATCH_LENGTH, N_BATCH);

    /* Create DMatrix from iterator.  During training, some cache files with the
     * prefix "cache-" will be generated in current directory */
    char config[] = "{\"missing\": NaN, \"cache_prefix\": \"cache\"}";
    DMatrixHandle Xy;
    XGDMatrixCreateFromCallback(&iter, iter._proxy, data_iterator_reset, data_iterator_next, config, &Xy);

    train_model(Xy);

    XGDMatrixFree(Xy);

    data_iterator_free(&iter);
    return 0;
}