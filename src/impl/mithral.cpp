//
//  mithral.cpp
//  Bolt
//
//  Created by DB on 12/3/19.
//  Copyright © 2019 D Blalock. All rights reserved.
//

#include "mithral.hpp"


// ================================================================ encode

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

void zip_bolt_colmajor(const uint8_t* codes_in, int64_t nrows, uint32_t ncodebooks, uint8_t* codes_out)
{
    if (ncodebooks % 8 == 0) 
    {
        zip_bolt_colmajor<8>(codes_in, nrows, ncodebooks, codes_out); return;
    }
    if (ncodebooks % 4 == 0) 
    {
        zip_bolt_colmajor<4>(codes_in, nrows, ncodebooks, codes_out); return;
    }
    zip_bolt_colmajor<2>(codes_in, nrows, ncodebooks, codes_out);
}

// ================================================================ lut

void dense_lut_f32_fused(const float* Q, int nrows, int ncols, int ncodebooks, const float* centroids, float*__restrict__ out_offsets, float& out_offset_sum, float& out_scale, float*__restrict__ out)
{
    static constexpr int codebook_tile_sz = 2;
    static constexpr int row_tile_sz = 2;
    dense_lut_f32_fused<codebook_tile_sz, row_tile_sz>(Q, nrows, ncols, ncodebooks, centroids, out_offsets, out_offset_sum, out_scale, out);
}


void mithral_lut_dense(const float* Q, int nrows, int ncols, int ncodebooks, const float* centroids, float& out_offset_sum, float& out_scale, float*__restrict__ tmp_lut_f32, uint8_t* out)
{
    float tmp_offsets[ncodebooks];
    dense_lut_f32_fused(Q, nrows, ncols, ncodebooks, centroids, tmp_offsets, out_offset_sum, out_scale, tmp_lut_f32);
    quantize_luts(tmp_lut_f32, nrows, ncodebooks, tmp_offsets, out_scale, out);
}

// ================================================================ scan

void mithral_scan(const uint8_t* codes, int64_t nblocks, int ncodebooks,
                  int noutputs, const uint8_t* luts, uint8_t* dists_out)
{
    mithral_scan<16, 2>(codes, nblocks, ncodebooks, noutputs, luts, dists_out);
}

// ================================================================== profile matmul

float profile_mithral(ColMatrix<float> &X, ColMatrix<float> &Q, int nbytes, bool create_lut = false) 
{
    int N = X.rows();
    int D = X.cols();
    int M = Q.cols();

    static constexpr int nsplits_per_codebook = 4;
    static constexpr int group_id_nbits = 4;
    static constexpr int max_ngroups = 1 << group_id_nbits;
    int ncodebooks = 2 * nbytes;
    int total_nsplits = ncodebooks * nsplits_per_codebook;
    int ncodebooks_pq = nbytes;
    static constexpr int ncentroids = 16;
    static constexpr int lut_sz = 16;
    static constexpr int scan_block_nrows = 32;
    auto nblocks = N / scan_block_nrows;

    ColMatrix<uint8_t> tmp_codes(N, ncodebooks); 
    tmp_codes.setRandom();

    ColMatrix<uint8_t> codes(N, ncodebooks); 
    codes.setRandom();
   
    float out_offset_sum;
    float out_scale;
    ColMatrix<uint16_t> out_mat(N, M);

    RowVector<uint32_t> splitdims_(total_nsplits);
    splitdims_.setRandom();
    RowVector<uint32_t> splitdims = splitdims_.unaryExpr(
        [=](const uint32_t x) { return x % D; });
    // RowVector<uint32_t> splitdims(total_nsplits); splitdims.setZero(); // TODO rm
    ColMatrix<int8_t> splitvals(max_ngroups, total_nsplits);
    splitvals.setRandom();
    RowVector<float> scales(MAX(D, total_nsplits)); // v2 needs D of these
    scales.setRandom();

    RowVector<float> offsets(MAX(D, total_nsplits)); // v2 needs D of these
    offsets.setRandom();

    RowMatrix<uint8_t> luts(N, ncodebooks * lut_sz);
    luts.setRandom();

    ColMatrix<float> centroids(ncentroids * ncodebooks, D); 
    centroids.setRandom();

    RowMatrix<float> tmp_luts_f32(N, ncodebooks * lut_sz);
    tmp_luts_f32.setRandom();

    // Start Timer
    auto begin = std::chrono::high_resolution_clock::now(); 

    // Encode
    mithral_encode(X.data(), N, D, splitdims.data(), splitvals.data(), scales.data(), offsets.data(), ncodebooks, tmp_codes.data());
    zip_bolt_colmajor(tmp_codes.data(), N, ncodebooks, codes.data());
    
    // LUT
    if(create_lut)  
    {
      mithral_lut_dense(Q.data(), M, D, ncodebooks, centroids.data(), out_offset_sum, out_scale, tmp_luts_f32.data(), luts.data());
    }

    // Scan
    mithral_scan(codes.data(), nblocks, ncodebooks, M, luts.data(), (uint8_t*)out_mat.data()); 
    
    // End Timer
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);

    return elapsed.count();
} 

float profile_matmul(ColMatrix<float> &X, ColMatrix<float> &Q)
{

    int N = X.rows();
    int D = X.cols();
    int M = Q.cols();

    ColMatrix<float> out_mat(N, M);

    // Start Timer
    auto begin = std::chrono::high_resolution_clock::now();

    // Matmul
    out_mat.noalias() = X * D;

    // End Timer
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    
    return elapsed.count() * 1e-9;

}
