package cat.mood;

import java.util.Arrays;

import static cat.mood.MatrixUtils.*;
import static java.lang.Math.abs;

public class C {
    static final double EPS = 1e-6;

    record Transformation(double[][] alpha, double[] beta) {}

    /**
     * Разрешение системы относительно диагональных переменных
     * @param matrix матрица коэффициентов
     * @param bias совбодные коэффициенты
     * @return матрица альфа и бета
     */
    static Transformation transform(double[][] matrix, double[] bias) {
        double[][] alpha = new double[matrix.length][matrix.length];
        double[] beta = new double[matrix.length];

        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix.length; ++j) {
                if (i != j) {
                    alpha[i][j] = - matrix[i][j] / matrix[i][i];
                }
            }
            beta[i] = bias[i] / matrix[i][i];
        }

        return new Transformation(alpha, beta);
    }

    /**
     * Метод итераций
     * @param matrix матрица коэффициентов
     * @param bias свободные коэффициенты
     * @param eps точность
     * @return вектор корней
     */
    public static double[] iteration(double[][] matrix, double[] bias, double eps) {
        Transformation t = transform(matrix, bias);

        double[] result = Arrays.copyOf(t.beta(), t.beta().length);
        double coef = lc(t.alpha()) / (1 - lc(t.alpha()));
        double epsilon = Double.MAX_VALUE;

        while (epsilon > eps) {
            double[] newResult = add(t.beta(), multiply(t.alpha(), result));
            epsilon = coef * lc(subtraction(newResult, result));

            result = newResult;
        }

        return result;
    }

    /**
     * Метод Зейделя
     * @param matrix матрица коэффициентов
     * @param bias свободные коэффициенты
     * @param eps точность
     * @return вектор корней
     */
    public static double[] seidel(double[][] matrix, double[] bias, double eps) {
        Transformation t = transform(matrix, bias);
        double[] result = Arrays.copyOf(t.beta(), t.beta().length);
        double coef = lc(t.alpha()) / (1 - lc(t.alpha()));
        double epsilon = Double.MAX_VALUE;

        while (epsilon > eps) {
            double[] newResult = Arrays.copyOf(t.beta(), t.beta().length);
            for (int i = 0; i < matrix.length; ++i) {
                for (int j = 0; j < matrix.length; ++j) {
                    if (j < i) {
                        newResult[i] += t.alpha[i][j] * newResult[j];
                    } else {
                        newResult[i] += t.alpha[i][j] * result[j];
                    }
                }
            }

            epsilon = coef * lc(subtraction(newResult, result));
            result = newResult;
        }

        return result;
    }

    /**
     * Проверка решения
     * @param matrix матрица коэффициентов
     * @param bias свободные коэффициенты
     * @param result вектор корней
     * @param eps точность
     * @return результат проверки (true/false)
     */
    public static boolean test(double[][] matrix, double[] bias, double[] result, double eps) {
        for (int i = 0; i < matrix.length; ++i) {
            double sum = 0;
            for (int j = 0; j < matrix.length; ++j) {
                sum += matrix[i][j] * result[j];
            }

            if (abs(sum - bias[i]) > eps) {
                return false;
            }
        }

        return true;
    }

    public static void main(String[] args) {
        double[][] matrix = new double[][] {
                {15, 0, 7, 5},
                {-3, -14, -6, 1},
                {-2, 9, 13, 2},
                {4, -1, 3, 9}
        };
        double[] bias = new double[] {176, -111, 74, 76};

        System.out.println("Метод итераций:");
        double[] result = iteration(matrix, bias, EPS);
        printVector(result);
        System.out.println("\nРезультат проверки: " + test(matrix, bias, result, 1e-2));

        System.out.println("Метод Зейделя:");
        result = seidel(matrix, bias, EPS);
        printVector(result);
        System.out.println("\nРезультат проверки: " + test(matrix, bias, result, 1e-2));
    }
}
