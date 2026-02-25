#include "matIO.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

// ===============================
// helpers
// ===============================
namespace {

// Read next non-empty, non-comment line (comment line begins with '%')
bool readDataLine(std::istream& in, std::string& line)
{
    while (std::getline(in, line))
    {
        if (line.empty()) continue;
        if (!line.empty() && line[0] == '%') continue;
        return true;
    }
    return false;
}

void expectHeader(std::istream& in)
{
    std::string header;
    if (!std::getline(in, header))
        throw std::runtime_error("Error: Empty file.");

    // Minimal check: must start with "%%MatrixMarket"
    if (header.rfind("%%MatrixMarket", 0) != 0)
        throw std::runtime_error("Error: Not a MatrixMarket file.");
}

void readSizeLine(std::istream& in, std::size_t& nrows, std::size_t& ncols, std::size_t& nnz)
{
    std::string line;
    if (!readDataLine(in, line))
        throw std::runtime_error("Error: Missing size line.");

    std::istringstream iss(line);
    if (!(iss >> nrows >> ncols >> nnz))
        throw std::runtime_error("Error: Invalid size line in MTX file.");
}

} // namespace

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
            if (A(i, j) != 0.0)
                ++nnz;

    out << "%%MatrixMarket matrix coordinate real general\n";
    out << N << " " << N << " " << nnz << "\n";

    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            if (A(i, j) != 0.0)
                out << (i + 1) << " " << (j + 1) << " " << A(i, j) << "\n";

    if (!out)
        throw std::runtime_error("Error: Failed while writing MTX file.");
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

    // Count nonzeros = (#nonzero diag) + (#stored off-diag values)
    std::size_t nnz = 0;
    for (std::size_t i = 0; i < N; ++i)
        if (d[i] != 0.0)
            ++nnz;
    nnz += va.size();

    out << "%%MatrixMarket matrix coordinate real general\n";
    out << N << " " << N << " " << nnz << "\n";

    // diagonal first
    for (std::size_t i = 0; i < N; ++i)
        if (d[i] != 0.0)
            out << (i + 1) << " " << (i + 1) << " " << d[i] << "\n";

    // off-diagonal (CRS)
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t p = rp[i]; p < rp[i + 1]; ++p)
            out << (i + 1) << " " << (ci[p] + 1) << " " << va[p] << "\n";

    if (!out)
        throw std::runtime_error("Error: Failed while writing MTX file.");
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

    expectHeader(in);

    std::size_t nrows = 0, ncols = 0, nnz = 0;
    readSizeLine(in, nrows, ncols, nnz);

    if (nrows != ncols)
        throw std::runtime_error("Error: Dim mismatching while reading (not square).");

    A = DenseSquareMatrixDouble(nrows);

    for (std::size_t k = 0; k < nnz; ++k)
    {
        std::size_t i = 0, j = 0;
        double val = 0.0;

        if (!(in >> i >> j >> val))
            throw std::runtime_error("Error: Failed reading an entry (i j val).");

        if (i == 0 || j == 0 || i > nrows || j > ncols)
            throw std::runtime_error("Error: Entry index out of range in MTX file.");

        A(i - 1, j - 1) = val;
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

    expectHeader(in);

    std::size_t nrows = 0, ncols = 0, nnz = 0;
    readSizeLine(in, nrows, ncols, nnz);

    if (nrows != ncols)
        throw std::runtime_error("Error: Dim mismatching while reading (not square).");

    A = SparseSquareMatrixCRSDouble(nrows);

    for (std::size_t k = 0; k < nnz; ++k)
    {
        std::size_t i = 0, j = 0;
        double val = 0.0;

        if (!(in >> i >> j >> val))
            throw std::runtime_error("Error: Failed reading an entry (i j val).");

        if (i == 0 || j == 0 || i > nrows || j > ncols)
            throw std::runtime_error("Error: Entry index out of range in MTX file.");

        A.addEntry(i - 1, j - 1, val);
    }

    A.finalize();
}

// ===============================
// Write Vector (MatrixMarket array)
// ===============================
void matIO::writeMTX(const std::string& filename,
                     const VectorDouble& v)
{
    std::ofstream out(filename);
    if (!out)
        throw std::runtime_error("Error: Cannot open file for writing.");

    const std::size_t N = v.size();

    out << "%%MatrixMarket matrix array real general\n";
    out << N << " " << 1 << "\n";

    for (std::size_t i = 0; i < N; ++i)
        out << v[i] << "\n";

    if (!out)
        throw std::runtime_error("Error: Failed while writing vector MTX file.");
}

// ===============================
// Read Vector (MatrixMarket array)
// ===============================
void matIO::readMTX(const std::string& filename,
                    VectorDouble& v)
{
    std::ifstream in(filename);
    if (!in)
        throw std::runtime_error("Error: Cannot open file for reading.");

    expectHeader(in);

    // For array format, the size line is: nrows ncols   (no nnz)
    std::string line;
    if (!readDataLine(in, line))
        throw std::runtime_error("Error: Missing size line.");

    std::istringstream iss(line);
    std::size_t nrows = 0, ncols = 0;
    if (!(iss >> nrows >> ncols))
        throw std::runtime_error("Error: Invalid size line in vector MTX file.");

    // allow N x 1 or 1 x N
    const bool col = (ncols == 1);
    const bool row = (nrows == 1);
    if (!(col || row))
        throw std::runtime_error("Error: Vector MTX must be N x 1 or 1 x N.");

    const std::size_t N = col ? nrows : ncols;

    v = VectorDouble(N);

    for (std::size_t i = 0; i < N; ++i)
    {
        double val;
        if (!(in >> val))
            throw std::runtime_error("Error: Not enough vector entries in MTX file.");
        v[i] = val;
    }
}