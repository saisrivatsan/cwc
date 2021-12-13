
#ifndef PUBLIC_INTERFACE_EIGEN_HPP
#define PUBLIC_INTERFACE_EIGEN_HPP

#include <Eigen/Dense>
#include <iostream>
#include <assert.h>
#include <stdint.h>
#include <sys/types.h>
#include <cmath>
#include <type_traits>
#include <limits>
#include "immintrin.h"
#include "avx_utils.hpp"
#include "eigen_utils.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
using Eigen::ArrayXXd;

using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::ArrayXf;
using Eigen::ArrayXXf;

using Eigen::MatrixXi;
using Eigen::VectorXi;
using Eigen::ArrayXi;
using Eigen::ArrayXXi;

MatrixXd createEigenMat();
VectorXd createEigenVect();
ArrayXXd createEigenArray();
ArrayXd createEigenArrayVect();

MatrixXf createEigenMatf();
VectorXf createEigenVectf();
ArrayXXf createEigenArrayf();
ArrayXf createEigenArrayVectf();

MatrixXi createEigenMati();
VectorXi createEigenVecti();
ArrayXXi createEigenArrayi();
ArrayXi createEigenArrayVecti();

int test_code();

void mithral_encode(const float* X, int64_t nrows, int ncols, const uint32_t* splitdims, const int8_t* all_splitvals, const float* scales, const float* offsets, int ncodebooks, uint8_t* out);
MatrixXf profile_encode(int N, int D, int nbytes);
#endif
