package cat.mood;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static cat.mood.MatrixUtils.*;
import static java.lang.Math.abs;
import static java.lang.Math.min;

public class E {
    static double L2(double[][] matrix, int index) {
        double sum = 0;
        for (int i = index; i < matrix.length; ++i) {
            sum += matrix[i][index] * matrix[i][index];
        }

        return Math.sqrt(sum);
    }

    public static PairMatrix QRDecomposition(double[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;

        if (n != m) {
            throw new IllegalArgumentException("Matrix does not have the same number of rows and columns");
        }
        double[][] E = new double[n][n];
        double[][] Q = new double[n][n];
        for (int i = 0; i < n; ++i) {
            E[i][i] = 1;
            Q[i][i] = 1;
        }


        double[][] A = copy2DArray(matrix);
        for (int i = 0; i < n; ++i) {
            double[][] v = new double[1][n];
            v[0][i] = A[i][i] + Math.signum(A[i][i]) * L2(A, i);
            for (int j = i + 1; j < n; ++j) {
                v[0][j] = A[j][i];
            }

            double[][] H = subtraction(E, multiplyByNumber(multiply(transpose(v), v), 2 / scalarMultiply(v[0], v[0])));
            A = multiply(H, A);
            Q = multiply(Q, H);
        }

        return new PairMatrix(Q, A);
    }

    static double sumUnderDiagonal(double[][] matrix) {
        double sum = 0;
        int n = matrix.length;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                sum += matrix[i][j] * matrix[i][j];
            }
        }

        return Math.sqrt(sum);
    }

    public static List<Complex> eigenvalues(double[][] matrix, double eps) {
        int n = matrix.length;
        int m = matrix[0].length;

        if (n != m) {
            throw new IllegalArgumentException("Matrix does not have the same number of rows and columns");
        }

        double[][] A = copy2DArray(matrix);
        List<Complex> eigenvalues = new ArrayList<>();

        int iters = 0;
        while (iters < 10) {
//        while (iters < 100000) {
            var decomposition = QRDecomposition(A);
            A = multiply(decomposition.second, decomposition.first);
            ++iters;
            printMatrix(A);
            System.out.println();
        }


        int i = 0;
        while (i < n) {
            if (i < n - 1 && abs(A[i + 1][i]) > eps) {
                handleComplexBlock(A, eigenvalues, i, eps);
                i += 2;
            } else {
                eigenvalues.add(new Complex(matrix[i][i], 0));
                ++i;
            }
        }

        System.out.println("Количество итераций: " + iters);
        System.out.println("Матрица A: ");
        printMatrix(A);

        return eigenvalues;
    }

    public static Complex[] solveQuadratic(double a, double b, double c) {
        double D = b * b - 4 * a * c;
        Complex[] roots = new Complex[2];

        if (D >= 0) {  // Вещественные корни
            roots[0] = new Complex((-b + Math.sqrt(D)) / (2 * a), 0);
            roots[1] = new Complex((-b - Math.sqrt(D)) / (2 * a), 0);
        } else {  // Комплексные корни
            double realPart = -b / (2 * a);
            double imagPart = Math.sqrt(-D) / (2 * a);
            roots[0] = new Complex(realPart, imagPart);
            roots[1] = new Complex(realPart, -imagPart);
        }
        return roots;
    }

    static boolean checkConvergence(double[][] matrix, List<Complex> eigenvalues, double eps) {
        int i = 0;
        int n = matrix.length;
        List<Complex> previous = List.copyOf(eigenvalues);
//        eigenvalues.clear();
        while (i < n) {
            if (i < n - 1 && abs(matrix[i + 1][i]) > eps) {
                if (!handleComplexBlock(matrix, eigenvalues, i, eps)) {
                    return false;
                }
                i += 2;
            } else {
                eigenvalues.add(new Complex(matrix[i][i], 0));
                ++i;
            }
        }
//
//        if (eigenvalues.size() != previous.size()) {
//            return false;
//        }
//
//        for (int j = 0; j < eigenvalues.size(); ++j) {
//            Complex sub = Complex.subtraction(eigenvalues.get(j), previous.get(j));
//            if (abs(sub.real) > eps || abs(sub.imaginary) > eps) {
//                return false;
//            }
//        }

        return true;
    }

    static boolean handleComplexBlock(double[][] matrix, List<Complex> eigenvalues, int i, double eps) {
        double a = matrix[i][i], b = matrix[i][i + 1], c = matrix[i + 1][i], d = matrix[i + 1][i + 1];
        Complex[] result = solveQuadratic(1, -(a + d), a * d - b * c);

        for (Complex root : result) {
            if (abs(root.real - matrix[i][i]) > eps && abs(root.imaginary) < eps) {
                return false;
            }
        }

        eigenvalues.addAll(Arrays.asList(result));
        return true;
    }

    public static void main(String[] args) {
        double[][] matrix = {
                {0, -1, 3},
                {-1, 6, -3},
                {-8, 4, 2}
        };
//        double[][] matrix = {
//                {1, 3, 1},
//                {1, 1, 4},
//                {4, 3, 1}
//        };

//        PairMatrix result = QRDecomposition(matrix);
//        printMatrix(result.first);
//        System.out.println();
//        printMatrix(result.second);
//        System.out.println();
//        printMatrix(multiply(result.first, result.second));

        System.out.println();
        var eigen = eigenvalues(matrix, 0.01);
        System.out.println(eigen.toString());
    }
}
