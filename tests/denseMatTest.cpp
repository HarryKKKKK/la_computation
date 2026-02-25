#include <iostream>
#include <cmath>
#include <stdexcept>

#include "VectorDouble.hpp"
#include "DenseSquareMatrixDouble.hpp"
#include "LinearSystemDense.hpp"

static void expect_near(double a, double b, double tol, const char* msg)
{
    if (std::abs(a - b) > tol) {
        std::cerr << "[FAIL] " << msg << " | got " << a << " expected " << b
                  << " (tol=" << tol << ")\n";
        std::exit(1);
    }
}

static void expect_true(bool cond, const char* msg)
{
    if (!cond) {
        std::cerr << "[FAIL] " << msg << "\n";
        std::exit(1);
    }
}

static void expect_false(bool cond, const char* msg)
{
    if (cond) {
        std::cerr << "[FAIL] " << msg << "\n";
        std::exit(1);
    }
}

static void test_vector_basic()
{
    std::cout << "Running test_vector_basic...\n";

    VectorDouble v(3);
    v[0] = 3.0;
    v[1] = 4.0;
    v[2] = 0.0;

    expect_near(v.norm_n(2), 5.0, 1e-12, "Vector norm2 should be 5");
    expect_near(v.normInf(), 4.0, 1e-12, "Vector normInf should be 4");

    VectorDouble w(3);
    w[0] = 1.0; w[1] = 2.0; w[2] = 3.0;

    VectorDouble a = v + w;
    expect_near(a[0], 4.0, 1e-12, "v+w[0]");
    expect_near(a[1], 6.0, 1e-12, "v+w[1]");
    expect_near(a[2], 3.0, 1e-12, "v+w[2]");

    VectorDouble s = w * 2.0;
    expect_near(s[0], 2.0, 1e-12, "w*2[0]");
    expect_near(s[1], 4.0, 1e-12, "w*2[1]");
    expect_near(s[2], 6.0, 1e-12, "w*2[2]");

    std::cout << "  OK\n";
}

static void test_dense_identity_mv()
{
    std::cout << "Running test_dense_identity_mv...\n";

    DenseSquareMatrixDouble A(4);
    VectorDouble x(4);

    for (std::size_t i = 0; i < 4; ++i) {
        A(i, i) = 1.0;
        x[i] = static_cast<double>(i + 1);
    }

    VectorDouble b = A * x;

    for (std::size_t i = 0; i < 4; ++i) {
        expect_near(b[i], x[i], 1e-12, "Identity matrix: A*x should equal x");
    }

    std::cout << "  OK\n";
}

static void test_dense_diagonal_mv()
{
    std::cout << "Running test_dense_diagonal_mv...\n";

    DenseSquareMatrixDouble A(3);
    VectorDouble x(3);

    A(0,0) = 2.0;
    A(1,1) = 3.0;
    A(2,2) = 4.0;

    x[0] = 1.0;
    x[1] = 2.0;
    x[2] = 3.0;

    VectorDouble b = A * x;

    expect_near(b[0], 2.0, 1e-12, "Diagonal MV b0");
    expect_near(b[1], 6.0, 1e-12, "Diagonal MV b1");
    expect_near(b[2], 12.0, 1e-12, "Diagonal MV b2");

    std::cout << "  OK\n";
}

static void test_dense_matrix_add_sub_scalar()
{
    std::cout << "Running test_dense_matrix_add_sub_scalar...\n";

    DenseSquareMatrixDouble A(2), B(2);

    // A = [1 2; 3 4]
    A(0,0)=1; A(0,1)=2;
    A(1,0)=3; A(1,1)=4;

    // B = [5 6; 7 8]
    B(0,0)=5; B(0,1)=6;
    B(1,0)=7; B(1,1)=8;

    DenseSquareMatrixDouble C = A + B;
    expect_near(C(0,0), 6, 1e-12, "A+B(0,0)");
    expect_near(C(1,1), 12, 1e-12, "A+B(1,1)");

    DenseSquareMatrixDouble D = B - A;
    expect_near(D(0,0), 4, 1e-12, "B-A(0,0)");
    expect_near(D(1,1), 4, 1e-12, "B-A(1,1)");

    DenseSquareMatrixDouble E = A * 2.0;
    expect_near(E(0,0), 2, 1e-12, "A*2(0,0)");
    expect_near(E(1,1), 8, 1e-12, "A*2(1,1)");

    std::cout << "  OK\n";
}

static void test_dense_matrix_matrix_mult()
{
    std::cout << "Running test_dense_matrix_matrix_mult...\n";

    DenseSquareMatrixDouble A(2), B(2);

    // A = [1 2; 3 4]
    A(0,0)=1; A(0,1)=2;
    A(1,0)=3; A(1,1)=4;

    // B = [5 6; 7 8]
    B(0,0)=5; B(0,1)=6;
    B(1,0)=7; B(1,1)=8;

    DenseSquareMatrixDouble C = A * B;

    // Expected:
    // [1*5+2*7, 1*6+2*8] = [19, 22]
    // [3*5+4*7, 3*6+4*8] = [43, 50]
    expect_near(C(0,0), 19, 1e-12, "A*B(0,0)");
    expect_near(C(0,1), 22, 1e-12, "A*B(0,1)");
    expect_near(C(1,0), 43, 1e-12, "A*B(1,0)");
    expect_near(C(1,1), 50, 1e-12, "A*B(1,1)");

    std::cout << "  OK\n";
}

static void test_linear_system_multiply_residual()
{
    std::cout << "Running test_linear_system_multiply_residual...\n";

    DenseSquareMatrixDouble A(3);
    VectorDouble x(3);
    VectorDouble b(3);

    // Identity matrix
    for (std::size_t i = 0; i < 3; ++i) {
        A(i,i) = 1.0;
        x[i] = static_cast<double>(i + 1);
        b[i] = 0.0;
    }

    LinearSystemDense sys(std::move(A), std::move(x), std::move(b));

    sys.multiply(); // b = A*x, so b should become [1,2,3]

    expect_near(sys.b()[0], 1.0, 1e-12, "sys.b()[0] after multiply");
    expect_near(sys.b()[1], 2.0, 1e-12, "sys.b()[1] after multiply");
    expect_near(sys.b()[2], 3.0, 1e-12, "sys.b()[2] after multiply");

    VectorDouble r = sys.residual();
    expect_near(r.normInf(), 0.0, 1e-12, "Residual should be zero for consistent identity system");

    std::cout << "  OK\n";
}

static void test_symmetry_and_diag_dominance()
{
    std::cout << "Running test_symmetry_and_diag_dominance...\n";

    // Symmetric + diagonally dominant matrix (tridiagonal)
    DenseSquareMatrixDouble A(3);
    VectorDouble x(3);
    VectorDouble b(3);

    // [ 4 -1  0
    //  -1  4 -1
    //   0 -1  4 ]
    A(0,0)=4; A(0,1)=-1;
    A(1,0)=-1; A(1,1)=4; A(1,2)=-1;
    A(2,1)=-1; A(2,2)=4;

    LinearSystemDense sys(std::move(A), std::move(x), std::move(b));

    expect_true(sys.isSymmetric(), "Matrix should be symmetric");
    expect_true(sys.isDiagonallyDominant(), "Matrix should be diagonally dominant");

    // Now make it asymmetric
    sys.A()(0,2) = 2.0; // but A(2,0) = 0
    expect_false(sys.isSymmetric(), "Matrix should NOT be symmetric after modification");

    std::cout << "  OK\n";
}

int main()
{
    try {
        test_vector_basic();
        test_dense_identity_mv();
        test_dense_diagonal_mv();
        test_dense_matrix_add_sub_scalar();
        test_dense_matrix_matrix_mult();
        test_linear_system_multiply_residual();
        test_symmetry_and_diag_dominance();

        std::cout << "\nAll tests PASSED\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] " << e.what() << "\n";
        return 1;
    }
}
