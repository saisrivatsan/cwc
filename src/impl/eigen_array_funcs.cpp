
#include "public_interface_eigen.hpp"

// TODO just instantiate two template funcs to avoid dup code

MatrixXd createEigenMat() {
	int nrows = 4;
	int ncols = 3;
	MatrixXd M(nrows, ncols);
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			M(i, j) = i * ncols + j; // 0-11 in row-major order
		}
	}
	return M;
}

VectorXd createEigenVect() {
	int nrows = 4;
	VectorXd v(nrows);
	for (int i = 0; i < nrows; i++) {
		v(i) = i;
	}
	return v;
}

ArrayXXd createEigenArray() {
	int nrows = 4;
	int ncols = 3;
	ArrayXXd A(nrows, ncols);
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			A(i, j) = i * ncols + j; // 0-11 in row-major order
		}
	}
	return A;
}

ArrayXd createEigenArrayVect() {
	int nrows = 4;
	ArrayXd v(nrows);
	for (int i = 0; i < nrows; i++) {
		v(i) = i;
	}
	return v;
}

// ------------------------------------------------ floats

MatrixXf createEigenMatf() {
	int nrows = 4;
	int ncols = 3;
	MatrixXf M(nrows, ncols);
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			M(i, j) = i * ncols + j; // 0-11 in row-major order
		}
	}
	return M;
}
VectorXf createEigenVectf() {
	int nrows = 4;
	VectorXf v(nrows);
	for (int i = 0; i < nrows; i++) {
		v(i) = i;
	}
	return v;
}

ArrayXXf createEigenArrayf() {
	int nrows = 4;
	int ncols = 3;
	ArrayXXf M(nrows, ncols);
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			M(i, j) = i * ncols + j; // 0-11 in row-major order
		}
	}
	return M;
}

ArrayXf createEigenArrayVectf() {
	int nrows = 4;
	ArrayXf v(nrows);
	for (int i = 0; i < nrows; i++) {
		v(i) = i;
	}
	return v;
}

// ------------------------------------------------ ints

MatrixXi createEigenMati() {
	int nrows = 4;
	int ncols = 3;
	MatrixXi M(nrows, ncols);
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			M(i, j) = i * ncols + j; // 0-11 in row-major order
		}
	}
	return M;
}
VectorXi createEigenVecti() {
	int nrows = 4;
	VectorXi v(nrows);
	for (int i = 0; i < nrows; i++) {
		v(i) = i;
	}
	return v;
}

ArrayXXi createEigenArrayi() {
	int nrows = 4;
	int ncols = 3;
	ArrayXXi M(nrows, ncols);
	for (int i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++) {
			M(i, j) = i * ncols + j; // 0-11 in row-major order
		}
	}
	return M;
}

ArrayXi createEigenArrayVecti() {
	int nrows = 4;
	ArrayXi v(nrows);
	for (int i = 0; i < nrows; i++) {
		v(i) = i;
	}
	return v;
}

int test_code()
{
  return 0;
}

/*
void mithral_encode(const float* X, int64_t nrows, int ncols, const uint32_t* splitdims, const int8_t* all_splitvals, const float* scales, const float* offsets, int ncodebooks, uint8_t* out)
{
    static constexpr bool DeferPerm = true;
    static constexpr int block_nrows = 32;
    static constexpr int nsplits_per_codebook = 4;
    static constexpr int vals_per_split = 1 << nsplits_per_codebook; // 16
    const int64_t nblocks = ceil(nrows / (double)block_nrows);
    assert(nrows % block_nrows == 0); // TODO remove this constraint

    // sanity check splits
    auto total_nsplits = ncodebooks * nsplits_per_codebook;
    auto maxdim = splitdims[0];
    auto mindim = splitdims[0];
    for (int i = 1; i < total_nsplits; i++) {
        maxdim = MAX(maxdim, splitdims[i]);
        mindim = MIN(maxdim, splitdims[i]);
    }
    assert(mindim >= 0);
    assert(maxdim < ncols);

    size_t x_col_stride = nrows;
    size_t out_col_stride = nrows;
    const float* x_ptrs[nsplits_per_codebook];
    __m256i current_vsplitval_luts[nsplits_per_codebook];
    __m256 current_vscales[nsplits_per_codebook];
    __m256 current_voffsets[nsplits_per_codebook];

    int split_idx = 0;
    for (int c = 0; c < ncodebooks; c++) {
        // compute input and output column starts
        auto out_ptr = out + (out_col_stride * c);
        for (int s = 0; s < nsplits_per_codebook; s++) {
            auto splitdim = splitdims[split_idx + s];
            x_ptrs[s] = X + (x_col_stride * splitdim);
            auto splitvals_ptr = all_splitvals + (vals_per_split * split_idx);
            current_vsplitval_luts[s] = _mm256_broadcastsi128_si256(
                load_si128i((const __m128i*)splitvals_ptr));
            current_vscales[s] = _mm256_set1_ps(scales[split_idx + s]);
            current_voffsets[s] = _mm256_set1_ps(offsets[split_idx + s]);
        }
        split_idx += nsplits_per_codebook;

        for (int b = 0; b < nblocks; b++) { // for each block
            __m256i codes = _mm256_setzero_si256();
            #pragma unroll
            for (int s = 0; s < nsplits_per_codebook; s++) {
                auto vscales = current_vscales[s];
                auto voffsets = current_voffsets[s];
                // auto voffsets = _mm256_setzero_si256();
                auto vsplitvals_lut = current_vsplitval_luts[s];
                auto vsplitvals = _mm256_shuffle_epi8(
                        vsplitvals_lut, codes); // codes = group_ids

                auto x_ptr = x_ptrs[s];
                x_ptrs[s] += block_nrows;

                // true = signed saturation; better because cmp instructions
                // exist for epi8 but not epu8
                auto x_i8 = load_4xf32_as_32xepi8_or_epu8<true, !DeferPerm>(
                    // x_ptr, vscales);
                    x_ptr, vscales, voffsets);

                auto masks = _mm256_cmpgt_epi8(x_i8, vsplitvals);
                // map -1 -> 1; 0 stays the same
                auto masks_0_or_1 = _mm256_sign_epi8(masks, masks);

                if (s > 0) {
                    // shift left by multiplying by 2, by adding to itself
                    codes = _mm256_add_epi8(codes, codes);
                }

                // OR in new low bit
                codes = _mm256_or_si256(codes, masks_0_or_1);
            }
            if (DeferPerm) {
                codes = _mm256_permutevar8x32_epi32(
                    codes, _mm256_setr_epi32(0,4, 1,5, 2,6, 3,7));
            }
            _mm256_storeu_si256((__m256i*)out_ptr, codes);
            out_ptr += block_nrows;
        }
    }
}


void mithral_scan(const uint8_t* codes, int64_t nblocks, int ncodebooks, int noutputs, const uint8_t* luts, uint8_t* dists_out)
{
    mithral_scan<16, 2>(codes, nblocks, ncodebooks, noutputs, luts, dists_out);
}



MatrixXf profile_encode(int N, int D, int nbytes) 
{
    static constexpr int nsplits_per_codebook = 4;
    static constexpr int group_id_nbits = 4;
    static constexpr int max_ngroups = 1 << group_id_nbits;
    int ncodebooks = 2 * nbytes;
    int total_nsplits = ncodebooks * nsplits_per_codebook;
    int ncodebooks_pq = nbytes;

    // shared
    MatrixXf X(N, D); X.setRandom();
    ColMatrix<uint8_t> out(N, ncodebooks); out.setRandom();

    // mithral-specific (it's a lot of stuff to handle different dtypes)
    RowVector<uint32_t> splitdims_(total_nsplits);
    splitdims_.setRandom();
    RowVector<uint32_t> splitdims = splitdims_.unaryExpr(
        [=](const uint32_t x) { return x % D; });
    // RowVector<uint32_t> splitdims(total_nsplits); splitdims.setZero(); // TODO rm
    ColMatrix<int8_t> all_splitvals(max_ngroups, total_nsplits);
    all_splitvals.setRandom();
    RowVector<float> scales(MAX(D, total_nsplits)); // v2 needs D of these
    scales.setRandom();
    RowVector<float> offsets(MAX(D, total_nsplits)); // v2 needs D of these
    offsets.setRandom();
    ColMatrix<int8_t> X_i8(N, D); X_i8.setRandom();
    ColMatrix<int16_t> X_i16(N, D); X_i16.setRandom();
    RowVector<int16_t> offsets_i16(total_nsplits); offsets_i16.setRandom();
    RowVector<uint8_t> shifts(total_nsplits); shifts.setRandom();

    mithral_encode(X.data(), N, D, splitdims.data(), all_splitvals.data(), scales.data(), offsets.data(), ncodebooks, out.data());
    
	  return X;
} */
