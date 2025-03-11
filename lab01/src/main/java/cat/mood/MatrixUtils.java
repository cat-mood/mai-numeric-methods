package cat.mood;

import java.util.Arrays;
import java.util.Locale;

import static java.lang.Math.abs;

public class MatrixUtils {
    public static final String PRECISION = "%.4f";
    public static final Locale LOCALE = Locale.US;

    /**
     * Напечатать матрицу
     * @param matrix матрица
     */
    public static void printMatrix(double[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                System.out.format(LOCALE, PRECISION, matrix[i][j]);
                System.out.print(" ");
            }
            System.out.println();
        }
    }

    /**
     * Напечатать вектор
     * @param vector вектор
     */
    public static void printVector(double[] vector) {
        int n = vector.length;

        for (int i = 0; i < n; ++i) {
            System.out.format(LOCALE, PRECISION, vector[i]);
            System.out.print(" ");
        }
    }

    /**
     * Получить копию матрицы
     * @param matrix матрица
     * @return копия matrix
     */
    public static double[][] copy2DArray(double[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;

        double[][] newMatrix = new double[n][m];

        for (int i = 0; i < n; ++i) {
            newMatrix[i] = Arrays.copyOf(matrix[i], m);
        }

        return newMatrix;
    }

    /**
     * Возвращает результат перемножения двух матриц
     * @param matrix1 первая матрица
     * @param matrix2 вторая матрица
     * @return произведение
     */
    public static double[][] multiply(double[][] matrix1, double[][] matrix2) {
        int p = matrix1.length;
        int q = matrix1[0].length;
        int q1 = matrix2.length;
        int r = matrix2[0].length;

        if (q != q1) {
            throw new IllegalArgumentException("Matrix lengths do not match");
        }

        double[][] result = new double[p][r];

        for (int i = 0; i < p; ++i) {
            for (int j = 0; j < r; ++j) {
                for (int k = 0; k < q; ++k) {
                    result[i][j] += matrix1[i][k] * matrix2[k][j];
                }
            }
        }

        return result;
    }

    /**
     * Возвращает результат перемножения матрицы на вектор
     * @param matrix матрица
     * @param vector вектор
     * @return произведение
     */
    public static double[] multiply(double[][] matrix, double[] vector) {
        double[][] matrixVector = new double[vector.length][1];
        for (int i = 0; i < vector.length; ++i) {
            matrixVector[i][0] = vector[i];
        }

        double[][] result = multiply(matrix, matrixVector);
        double[] resultVector = new double[result.length];
        for (int i = 0; i < result.length; ++i) {
            resultVector[i] = result[i][0];
        }

        return resultVector;
    }

    /**
     * l-c норма
     * @param vector вектор
     * @return значение нормы
     */
    public static double lc(double[] vector) {
        int n = vector.length;
        double max = vector[0];

        for (int i = 1; i < n; ++i) {
            if (abs(vector[i]) > max) {
                max = abs(vector[i]);
            }
        }

        return max;
    }

    /**
     * l-c норма
     * @param matrix матрица
     * @return значение нормы
     */
    public static double lc(double[][] matrix) {
        int n = matrix.length;
        int m = matrix[0].length;
        double max = - Double.MAX_VALUE;

        for (int i = 0; i < n; ++i) {
            double sum = 0;
            for (int j = 0; j < m; ++j) {
                sum += abs(matrix[i][j]);
            }
            if (sum > max) {
                max = sum;
            }
        }

        return max;
    }

    /**
     * Вычесть два вектора
     * @param vector1 уменьшаемое
     * @param vector2 вычитаемое
     * @return разность
     */
    public static double[] subtraction(double[] vector1, double[] vector2) {
        int n = vector1.length;
        int m = vector2.length;

        if (n != m) {
            throw new IllegalArgumentException("Vector lengths do not match");
        }

        double[] result = new double[n];

        for (int i = 0; i < n; ++i) {
            result[i] = vector1[i] - vector2[i];
        }

        return result;
    }

    /**
     * Сложение двух векторов
     * @param vector1 первое слагаемое
     * @param vector2 второе слагаемое
     * @return сумма
     */
    public static double[] add(double[] vector1, double[] vector2) {
        int n = vector1.length;
        int m = vector2.length;

        if (n != m) {
            throw new IllegalArgumentException("Vector lengths do not match");
        }

        double[] result = new double[n];
        for (int i = 0; i < n; ++i) {
            result[i] = vector1[i] + vector2[i];
        }

        return result;
    }
}
