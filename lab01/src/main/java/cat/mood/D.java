package cat.mood;

import static cat.mood.MatrixUtils.*;

public class D {
    static Pair<Integer, Integer> maxNotDiagonal(double[][] matrix) {
        int iMax = 0, jMax = 1;

        for (int i = 0; i < matrix.length; ++i) {
            for (int j = i + 1; j < matrix.length; ++j) {
                if (Math.abs(matrix[i][j]) > Math.abs(matrix[iMax][jMax])) {
                    iMax = i;
                    jMax = j;
                }
            }
        }

        return new Pair<>(iMax, jMax);
    }

    static boolean isSymmetrical(double[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;

        if (n != m) return false;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (matrix[i][j] != matrix[j][i]) return false;
            }
        }

        return true;
    }

    static PairMatrix jacobiRotation(double[][] matrix, double eps) {
        if (!isSymmetrical(matrix)) return null;
        int n = matrix.length;

        double[][] A = copy2DArray(matrix);
        double[][] resultU = new double[n][n];

        for (int i = 0; i < n; ++i) {
            resultU[i][i] = 1;
        }

        double sum = eps + 1;
        int iters = 0;
        while (sum > eps) {
            double[][] U = new double[n][n];
            var max = maxNotDiagonal(A);
            for (int i = 0; i < n; ++i) {
                U[i][i] = 1;
            }
            double phi;
            if (Math.abs(A[max.first][max.first] - A[max.second][max.second]) < eps) {
                phi = Math.PI / 4;
            } else {
                phi = 0.5 * Math.atan(2 * A[max.first][max.second] / (A[max.first][max.first] - A[max.second][max.second]));
            }
            U[max.first][max.first] = Math.cos(phi);
            U[max.first][max.second] = - Math.sin(phi);
            U[max.second][max.first] = Math.sin(phi);
            U[max.second][max.second] = Math.cos(phi);

            double[][] T = transpose(U);
            double[][] TA = multiplyMatrices(T, A);

            A = multiply(TA, U);

            sum = 0;

            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    sum += A[i][j] * A[i][j];
                }
            }
            sum = Math.sqrt(sum);
            ++iters;
            resultU = multiply(resultU, U);
        }
        System.out.println("Количество итераций: " + iters);

        return new PairMatrix(A, resultU);
    }

    public static void main(String[] args) {
        double[][] matrix = {
                {-8, 9, 6},
                {9, 9, 1},
                {6, 1, 8}
        };

        var result = jacobiRotation(matrix, 0.000001);
        System.out.println("Диагональная матрица:");
        printMatrix(result.first);
        System.out.println("Матрица собственных векторов:");
        printMatrix(result.second);
        System.out.println("Проверка на ортогональность:");
        double[][] transposed = transpose(result.second);
        for (int i = 0; i < result.second.length; ++i) {
            for (int j = i + 1; j < result.second.length; ++j) {
                double mult = scalarMultiply(transposed[i], transposed[i + 1]);
                System.out.print("(x" + i + ", x" + (j) + ") = " + mult);
                System.out.println();
            }
        }
    }
}
