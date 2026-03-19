#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <algorithm>

#include "VectorDouble.hpp"
#include "DenseSquareMatrixDouble.hpp"
#include "SparseSquareMatrixCRSDouble.hpp"

void populate_random_matrix(std::size_t N, double fill_ratio, 
                            DenseSquareMatrixDouble& Ad, 
                            SparseSquareMatrixCRSDouble& As) 
{
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_real_distribution<double> val_dist(-1.0, 1.0);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            if (dist(rng) < fill_ratio) {
                double v = val_dist(rng);
                Ad(i, j) = v;
                As.addEntry(i, j, v);
            }
        }
    }
    As.finalize();
}

int main() {
    const std::size_t N = 2000;
    const int iterations = 10;
    std::vector<double> fill_ratios = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};

    std::ofstream out("bench_sparsity_study.csv");
    out << "fill_ratio,storage,time_sec\n";

    VectorDouble x(N); 
    for(std::size_t i=0; i<N; ++i) x[i] = 1.0;

    for (double fr : fill_ratios) {
        DenseSquareMatrixDouble Ad(N);
        SparseSquareMatrixCRSDouble As(N);
        populate_random_matrix(N, fr, Ad, As);

        auto t0 = std::chrono::high_resolution_clock::now();
        for(int k=0; k<iterations; ++k) {
            auto y = Ad * x;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        out << fr << ",dense," << std::chrono::duration<double>(t1 - t0).count() / iterations << "\n";

        auto t2 = std::chrono::high_resolution_clock::now();
        for(int k=0; k<iterations; ++k) {
            auto y = As * x;
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        out << fr << ",sparse," << std::chrono::duration<double>(t3 - t2).count() / iterations << "\n";
    }

    out.close();
    return 0;
}