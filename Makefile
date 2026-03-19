# # ==============================
# # Compiler settings
# # ==============================
# CXX      = g++
# CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Iinclude

# # ==============================
# # Source files
# # ==============================
# SRC = \
# 	src/VectorDouble.cpp \
# 	src/DenseSquareMatrixDouble.cpp \
# 	src/LinearSystemDense.cpp \
# 	src/SparseSquareMatrixCRSDouble.cpp \
# 	src/LinearSystemSparse.cpp \
# 	src/matIO.cpp

# # ==============================
# # Targets
# # ==============================

# .PHONY: all dense sparse io performance clean

# all: dense sparse io performance

# dense:
# 	$(CXX) $(CXXFLAGS) $(SRC) tests/denseMatTest.cpp -o denseTest

# sparse:
# 	$(CXX) $(CXXFLAGS) $(SRC) tests/sparseMatTest.cpp -o sparseTest

# io:
# 	$(CXX) $(CXXFLAGS) $(SRC) tests/ioTest.cpp -o ioTest

# performance:
# 	$(CXX) $(CXXFLAGS) $(SRC) computational_performance.cpp -o performance

# # -------- Clean ---------------
# clean:
# 	rm -f denseTest sparseTest ioTest performance

CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++17
INCLUDES = -Iinclude
SRC_DIR = src
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)

TARGET = bench_sparsity
MAIN_SRC = dimension_efficiency.cpp

all: $(TARGET)

$(TARGET): $(MAIN_SRC) $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)