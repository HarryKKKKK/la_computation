#pragma once
#include <cstddef>
#include <string>

#include "DenseSquareMatrixDouble.hpp"
#include "SparseSquareMatrixCRSDouble.hpp"

namespace matIO {

// -------- Dense --------
void writeMTX(const std::string& filename,
              const DenseSquareMatrixDouble& A);

void readMTX(const std::string& filename,
             DenseSquareMatrixDouble& A);

// -------- Sparse --------
void writeMTX(const std::string& filename,
              const SparseSquareMatrixCRSDouble& A);

void readMTX(const std::string& filename,
             SparseSquareMatrixCRSDouble& A);

}