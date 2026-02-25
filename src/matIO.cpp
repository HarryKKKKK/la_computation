#include "matIO.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

// ===============================
// Write Dense Matrix
// ===============================
void matIO::writeMTX(const std::string& filename,
                     const DenseSquareMatrixDouble& A)
{
    std::ofstream out(filename);
    if (!out)
        throw std::runtime_error("Error: Cannot open file for writing.");

    const std::size_t N = A.size();

    // Count nonzeros
    std::size_t nnz = 0;
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            if (A(i,j) != 0.0)
                ++nnz;

    out << "%%MatrixMarket matrix coordinate real general\n";
    out << N << " " << N << " " << nnz << "\n";

    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            if (A(i,j) != 0.0)
                out << (i+1) << " " << (j+1) << " " << A(i,j) << "\n";
}

// ===============================
// Write Sparse Matrix
// ===============================
void matIO::writeMTX(const std::string& filename,
                     const SparseSquareMatrixCRSDouble& A)
{
    std::ofstream out(filename);
    if (!out)
        throw std::runtime_error("Error: Cannot open file for writing.");

    const std::size_t N = A.size();

    const auto& rp = A.rowPtr();
    const auto& ci = A.colInd();
    const auto& va = A.values();
    const auto& d  = A.diagonal();

    std::size_t nnz = 0;

    // Count nonzeros
    for (std::size_t i = 0; i < N; ++i)
        if (d[i] != 0.0)
            ++nnz;
    nnz += va.size();

    out << "%%MatrixMarket matrix coordinate real general\n";
    out << N << " " << N << " " << nnz << "\n";

    // diagonal
    for (std::size_t i = 0; i < N; ++i)
        if (d[i] != 0.0)
            out << (i+1) << " " << (i+1) << " " << d[i] << "\n";

    // off-diagonal
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t p = rp[i]; p < rp[i+1]; ++p)
            out << (i+1) << " "
                << (ci[p]+1) << " "
                << va[p] << "\n";
}

// ===============================
// Read Dense Matrix
// ===============================
void matIO::readMTX(const std::string& filename,
                    DenseSquareMatrixDouble& A)
{
    std::ifstream in(filename);
    if (!in)
        throw std::runtime_error("Error: Cannot open file for reading.");

    std::string line;

    std::getline(in, line);

    // skip comments
    do {
        std::getline(in, line);
    } while (line[0] == '%');

    std::istringstream iss(line);

    std::size_t nrows, ncols, nnz;
    iss >> nrows >> ncols >> nnz;

    if (nrows != ncols)
        throw std::runtime_error("Error: Dim mismatching while reading");

    A = DenseSquareMatrixDouble(nrows);

    for (std::size_t k = 0; k < nnz; ++k)
    {
        std::size_t i, j;
        double val;
        in >> i >> j >> val;

        A(i-1, j-1) = val;
    }
}

// ===============================
// Read Sparse Matrix
// ===============================
void matIO::readMTX(const std::string& filename,
                    SparseSquareMatrixCRSDouble& A)
{
    std::ifstream in(filename);
    if (!in)
        throw std::runtime_error("Error: Cannot open file for reading.");

    std::string line;

    std::getline(in, line);

    do {
        std::getline(in, line);
    } while (line[0] == '%');

    std::istringstream iss(line);

    std::size_t nrows, ncols, nnz;
    iss >> nrows >> ncols >> nnz;

    if (nrows != ncols)
        throw std::runtime_error("Error: Dim mismatching while reading");

    A = SparseSquareMatrixCRSDouble(nrows);

    for (std::size_t k = 0; k < nnz; ++k)
    {
        std::size_t i, j;
        double val;
        in >> i >> j >> val;

        A.addEntry(i-1, j-1, val);
    }

    A.finalize();
}