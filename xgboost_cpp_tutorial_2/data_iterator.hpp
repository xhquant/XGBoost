#ifndef XGBOOST_DATA_ITERATOR_HPP
#define XGBOOST_DATA_ITERATOR_HPP

#include <cstdlib>
#include <cstddef>
#include <cstring>

#include "xgboost/c_api.h"

struct data_iterator
{
    /* Data of each batch. */
    float **data;

    /* Labels of each batch */
    float **labels;

    /* Length of each batch. */
    size_t *lengths;

    /* Total number of batches. */
    size_t n;

    /* Current iteration. */
    size_t cur_it;

    /* Private fields */
    DMatrixHandle _proxy;
    char _array[128];
};


////////////////////////////////////////////////////////////////////////
/// \brief
/// \param self
/// \param batch_size
/// \param n_batches
inline
void data_iterator_init(data_iterator *self, size_t batch_size, size_t n_batches)
{
    std::cout << __FUNCTION__ << std::endl;

    self->n = n_batches;

    self->lengths = (size_t *) malloc(self->n * sizeof(size_t));
    for (size_t i = 0; i != self->n; ++i)
    {
        self->lengths[i] = batch_size;
    }

    self->data = (float **) malloc(self->n * sizeof(float *));

    self->labels = (float **) malloc(self->n * sizeof(float *));

    /* Generate some random data. */
    for (size_t i = 0; i < self->n; ++i)
    {
        self->data[i] = (float *) malloc(self->lengths[i] * sizeof(float));
        for (size_t j = 0; j < self->lengths[i]; ++j)
        {
            float x = (float) rand() / (float) (RAND_MAX);
            self->data[i][j] = x;
        }

        self->labels[i] = (float *) malloc(self->lengths[i] * sizeof(float));
        for (size_t j = 0; j < self->lengths[i]; ++j)
        {
            float y = (float) rand() / (float) (RAND_MAX);
            self->labels[i][j] = y;
        }
    }

    self->cur_it = 0;

    XGProxyDMatrixCreate(&self->_proxy);
}


////////////////////////////////////////////////////////////////////////
/// \brief
/// \param self
inline
void data_iterator_free(data_iterator *self)
{
    std::cout << __FUNCTION__ << std::endl;

    for (size_t i = 0; i < self->n; ++i)
    {
        free(self->data[i]);
        free(self->labels[i]);
    }

    free(self->data);
    free(self->lengths);
    free(self->labels);
    XGDMatrixFree(self->_proxy);
}


////////////////////////////////////////////////////////////////////////
/// \brief
/// \param handle
inline
int data_iterator_next(DataIterHandle handle)
{
    std::cout << __FUNCTION__ << std::endl;

    auto *self = (data_iterator *) (handle);
    if (self->cur_it == self->n)
    {
        self->cur_it = 0;
        return 0;  /* At end */
    }

    /* A JSON string encoding array interface (standard from numpy). */
    char array[] = "{\"data\": [%lu, false], \"shape\":[%lu, 1], \"typestr\": "
                   "\"<f4\", \"version\": 3}";
    memset(self->_array, '\0', sizeof(self->_array));
    sprintf(self->_array, array, (size_t) self->data[self->cur_it], self->lengths[self->cur_it]);

    XGProxyDMatrixSetDataDense(self->_proxy, self->_array);
    /* The data passed in the iterator must remain valid (not being freed until the next
     * iteration or reset) */
    XGDMatrixSetDenseInfo(self->_proxy, "label", self->labels[self->cur_it], self->lengths[self->cur_it], 1);

    self->cur_it++;
    return 1;  /* Continue. */
}


////////////////////////////////////////////////////////////////////////
/// \brief
/// \param handle
inline
void data_iterator_reset(DataIterHandle handle)
{
    std::cout << __FUNCTION__ << std::endl;

    auto *self = (data_iterator *) (handle);
    self->cur_it = 0;
}


////////////////////////////////////////////////////////////////////////
/// \brief
/// \param Xy
inline
void train_model(DMatrixHandle Xy)
{
    std::cout << __FUNCTION__ << std::endl;

    BoosterHandle booster;
    DMatrixHandle cache[] = {Xy};

    XGBoosterCreate(cache, 1, &booster);
    /* Use approx for external memory training. */
    XGBoosterSetParam(booster, "tree_method", "approx");
    XGBoosterSetParam(booster, "objective", "reg:squarederror");

    /* Start training. */
    const char *validation_names[1] = {"train"};
    const char *validation_result = NULL;
    size_t n_rounds = 10;
    for (size_t i = 0; i < n_rounds; ++i)
    {
        XGBoosterUpdateOneIter(booster, i, Xy);
        XGBoosterEvalOneIter(booster, i, cache, validation_names, 1, &validation_result);
        printf("%s\n", validation_result);
    }

    /* Save the model to a JSON file. */
    XGBoosterSaveModel(booster, "model.json");
    XGBoosterFree(booster);
}


#endif //XGBOOST_DATA_ITERATOR_HPP
