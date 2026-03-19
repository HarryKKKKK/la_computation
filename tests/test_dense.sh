#!/bin/bash

echo "Compiling..."
# g++ -O3 -fopenmp test_dense.cpp src/VectorDouble.cpp src/DenseSquareMatrixDouble.cpp src/LinearSystemDense.cpp -o bench_dense

OUTPUT="bench_dense_final.csv"
echo "operation,n,threads,execution_time_sec" > $OUTPUT

NS="16 64 256 1024 4096 10000"
THREADS="1 2 4 8"

for n in $NS; do
    echo "Running scale n=$n..."
    
    export OMP_NUM_THREADS=1
    ./bench_dense $n "nominal" >> $OUTPUT
    
    for p in $THREADS; do
        export OMP_NUM_THREADS=$p
        echo "  - Threads: $p"
        ./bench_dense $n "optimised" >> $OUTPUT
    done
done

echo "Done! Results saved to $OUTPUT"