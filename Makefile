# ==============================
# Compiler settings
# ==============================
CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Iinclude

# ==============================
# Source files
# ==============================
SRC = \
	src/VectorDouble.cpp \
	src/DenseSquareMatrixDouble.cpp \
	src/LinearSystemDense.cpp \
	src/SparseSquareMatrixCRSDouble.cpp \
	src/LinearSystemSparse.cpp \
	src/matIO.cpp

# ==============================
# Targets
# ==============================

.PHONY: all dense sparse io clean

all: dense sparse io

# -------- Dense test ----------
dense:
	$(CXX) $(CXXFLAGS) $(SRC) tests/denseMatTest.cpp -o denseTest

# -------- Sparse test ---------
sparse:
	$(CXX) $(CXXFLAGS) $(SRC) tests/sparseMatTest.cpp -o sparseTest

# -------- IO test -------------
io:
	$(CXX) $(CXXFLAGS) $(SRC) tests/ioTest.cpp -o ioTest

# -------- Clean ---------------
clean:
	rm -f denseTest sparseTest ioTest