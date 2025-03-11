package cat.mood;

import static cat.mood.MatrixUtils.*;
import static java.lang.Math.abs;

public class A {
    static final double EPS = 1e-6;

    /**
     * Привести матрицу к верхнему треугольному виду
     * @param matrix матрица
     * @param bias свободные коэффициенты
     * @return коэффициент определителя (1 или -1)
     */
    public static int transform(double[][] matrix, double[] bias) {
        int n = matrix.length;
        int detCoef = 1;

        int maxIndex = 0;
        for (int i = 0; i < n; ++i) {
            if (abs(matrix[i][0]) > matrix[maxIndex][0]) {
                maxIndex = i;
            }
        }

        if (maxIndex != 0) {
            double temp;
            for (int j = 0; j < n; ++j) {
                temp = matrix[0][j];
                matrix[0][j] = matrix[maxIndex][j];
                matrix[maxIndex][j] = temp;
            }
            temp = bias[0];
            bias[0] = bias[maxIndex];
            bias[maxIndex] = temp;
            detCoef *= -1;
        }

        for (int k = 0; k < n - 1; ++k) {
            for (int i = k + 1; i < n; ++i) {
                double coef = (abs(matrix[i][k]) < EPS) ? 0 : matrix[i][k] / matrix[k][k];
                for (int j = k; j < n; ++j) {
                    matrix[i][j] -= coef * matrix[k][j];
                }
                bias[i] -= coef * bias[k];
            }
        }

        return detCoef;
    }

    /**
     * Привести матрицу к верхнему треугольному виду
     * @param matrix матрица
     * @param bias свободные коэффициенты в виде матрицы
     * @return коэффициент определителя (1 или -1)
     */
    public static int transform(double[][] matrix, double[][] bias) {
        int n = matrix.length;
        int detCoef = 1;

        for (int k = 0; k < n - 1; ++k) {
            for (int i = k + 1; i < n; ++i) {
                double coef = (abs(matrix[i][k]) < EPS) ? 0 : matrix[i][k] / matrix[k][k];
                for (int j = 0; j < n; ++j) {
                    matrix[i][j] -= coef * matrix[k][j];
                    bias[i][j] -= coef * bias[k][j];
                }
            }
        }

        return detCoef;
    }

    /**
     * Найти определитель
     * @param matrix матрица
     * @param detCoef коэффициент определителя (1 или -1)
     * @return определитель
     * */
    public static double determinant(double[][] matrix, int detCoef) {
        int n = matrix.length;
        double determinant = detCoef;

        for (int i = 0; i < n; ++i) {
            determinant *= matrix[i][i];
        }

        return determinant;
    }

    /**
     * Решить СЛАУ
     * @param matrix верхняя треугольная матрица СЛАУ
     * @param bias свободные коэффициенты
     * @return корни СЛАУ
     */
    public static double[] solve(double[][] matrix, double[] bias) {
        int n = matrix.length;
        double[] result = new double[n];

        for (int i = n - 1; i >= 0; --i) {
            result[i] = bias[i];
            for (int j = i + 1; j < n; ++j) {
                result[i] -= matrix[i][j] * result[j];
            }
            result[i] /= matrix[i][i];
        }

        return result;
    }

    /**
     * Транспонировать матрицу
     * @param matrix матрица
     */
    public static void transpose(double[][] matrix) {
        int n = matrix.length;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                double temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
    }

    /**
     * Найти обратную матрицу
     * @param matrix матрица
     * @return обратная матрица
     */
    public static double[][] inverse(double[][] matrix) {
        int n = matrix.length;
        double[][] result = new double[n][n];
        double[][] identity = new double[n][n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                identity[i][j] = (i == j) ? 1 : 0;
            }
        }

        double[][] matrixCopy = copy2DArray(matrix);
        int detCoef = transform(matrixCopy, identity);
        double[] bias = new double[n];
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                bias[i] = identity[i][j];
            }
            result[j] = solve(matrixCopy, bias);
        }

        transpose(result);

        return result;
    }

    public static void main(String[] args) {
        double[][] matrix = {
                {-8, 5, 8, -6},
                {2, 7, -8, -1},
                {-5, -4, 1, -6},
                {5, -9, -2, 8}
        };
        double[] bias = {-144, 25, -21, 103};
        System.out.println("Обратная матрица:");
        double[][] inverse = inverse(matrix);
        printMatrix(inverse);
        System.out.println("A * A^(-1) =");
        printMatrix(multiply(matrix, inverse));
        int detCoef = transform(matrix, bias);
        double[] result = solve(matrix, bias);
        System.out.println("Решение СЛАУ:");
        printVector(result);
        System.out.println();
        System.out.print("Определитель: ");
        System.out.format(LOCALE, PRECISION, determinant(matrix, detCoef));
    }
}