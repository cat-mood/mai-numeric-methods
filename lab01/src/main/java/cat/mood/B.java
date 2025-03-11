package cat.mood;

import java.util.ArrayList;
import java.util.List;

import static cat.mood.MatrixUtils.printVector;
import static java.lang.Math.abs;

public class B {
    static final double EPS = 1e-6;

    /**
     * Решить систему методом прогонки
     * @param matrix матрица коэффициентов
     * @param bias свободные коэффициенты
     * @return вектор решений
     */
    public static double[] solve(double[][] matrix, double[] bias) {
        double[] result = new double[matrix.length];
        double[] P = new double[matrix.length];
        double[] Q = new double[matrix.length];

        P[0] = - matrix[0][2] / matrix[0][1];
        Q[0] = bias[0] / matrix[0][1];
        for (int i = 1; i < matrix.length; ++i) {
            P[i] = - matrix[i][2] / (matrix[i][1] + matrix[i][0] * P[i - 1]);
            Q[i] = (bias[i] - matrix[i][0] * Q[i - 1]) / (matrix[i][1] + matrix[i][0] * P[i - 1]);
        }

        result[matrix.length - 1] = Q[matrix.length - 1];
        for (int i = matrix.length - 2; i >= 0; --i) {
            result[i] = P[i] * result[i + 1] + Q[i];
        }

        return result;
    }

    /**
     * Протестировать решение
     * @param matrix матрица коэффициентов
     * @param bias свободные коэффициенты
     * @param result вектор решений
     * @return корректное/некорректное решение
     */
    public static boolean test(double[][] matrix, double[] bias, double[] result) {
        double lhs = matrix[0][1] * result[0] + matrix[0][2] * result[1];
        if (abs(lhs - bias[0]) >= EPS) {
            return false;
        }

        for (int i = 1; i < matrix.length - 1; ++i) {
            lhs = matrix[i][0] * result[i - 1] + matrix[i][1] * result[i] + matrix[i][2] * result[i + 1];
            if (abs(lhs - bias[i]) >= EPS) {
                return false;
            }
        }

        lhs = matrix[matrix.length - 1][0] * result[matrix.length - 2] + matrix[matrix.length - 1][1] * result[matrix.length - 1];

        if (abs(lhs - bias[matrix.length - 1]) >= EPS) {
            return false;
        }

        return true;
    }

    public static void main(String[] args) {
        double[][] matrix = new double[][] {
                {0, 10, -1},
                {-8, 16, 1},
                {6, -16, 6},
                {-8, 16, -5},
                {5, -13, 0}
        };

        double[] bias = new double[] {16, -110, 24, -3, 87};

        double[] result = solve(matrix, bias);

        System.out.println("Результат:");
        printVector(result);

        System.out.println("\nПроверка: " + test(matrix, bias, result));
    }
}
