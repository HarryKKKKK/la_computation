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

.PHONY: all dense sparse io performance clean

all: dense sparse io performance

dense:
	$(CXX) $(CXXFLAGS) $(SRC) tests/denseMatTest.cpp -o denseTest

sparse:
	$(CXX) $(CXXFLAGS) $(SRC) tests/sparseMatTest.cpp -o sparseTest

io:
	$(CXX) $(CXXFLAGS) $(SRC) tests/ioTest.cpp -o ioTest

performance:
	$(CXX) $(CXXFLAGS) $(SRC) computational_performance.cpp -o performance

# -------- Clean ---------------
clean:
	rm -f denseTest sparseTest ioTest performance