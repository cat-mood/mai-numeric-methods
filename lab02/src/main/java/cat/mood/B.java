package cat.mood;

import java.util.Arrays;
import java.util.function.Function;

public class B {
    public static void main(String[] args) {
        Function<double[], double[]> phi = x -> new double[]{
                Math.sqrt(2 * Math.log10(x[1]) + 1),
                (x[0] * x[0] + 1) / x[0]
        };

        double[] initialGuess = {1.5, 2};
        int maxIterations = 100;
        double eps = 1e-6;
        double checkRadius = 0.5;

        double[] solution = iterations(phi, initialGuess, maxIterations, eps, checkRadius);

        System.out.println("Метод итераций:");
        if (solution != null) {
            System.out.println("Решение найдено:");
            for (int i = 0; i < solution.length; i++) {
                System.out.printf("x%d = %.6f\n", i, solution[i]);
            }
        } else {
            System.out.println("Решение не найдено (нарушено условие сходимости).");
        }

        Function<double[], double[]> f = x -> new double[]{
                x[0] * x[0] - 2 * Math.log10(x[1]) - 1,
                x[0] * x[0] - x[0] * x[1] + 1
        };

        solution = newton(f, initialGuess, maxIterations, eps);
        System.out.println("Метод Ньютона:");
        System.out.println("Решение найдено:");
        for (int i = 0; i < solution.length; i++) {
            System.out.printf("x%d = %.6f\n", i, solution[i]);
        }
    }

    public static double[] iterations(
            Function<double[], double[]> phi,
            double[] initialGuess,
            int maxIterations,
            double eps,
            double checkRadius) {

        int n = initialGuess.length;
        double[] current = Arrays.copyOf(initialGuess, n);

        if (!checkConvergence(phi, current, checkRadius, eps)) {
            return null;
        }

        for (int iter = 0; iter < maxIterations; iter++) {
            double[] next = phi.apply(current);
            double error = 0;

            for (int i = 0; i < n; i++) {
                error = Math.max(error, Math.abs(next[i] - current[i]));
            }

            if (error < eps) {
                System.out.printf("Сходимость достигнута за %d итераций.\n", iter + 1);
                return next;
            }

            current = Arrays.copyOf(next, n);
        }

        System.out.println("Достигнуто максимальное число итераций.");
        return current;
    }

    // Проверка условия сходимости (||J||_inf < 1)
    public static boolean checkConvergence(
            Function<double[], double[]> phi,
            double[] point,
            double radius,
            double eps) {

        int n = point.length;
        double[][] testPoints = generateTestPoints(point, radius);

        for (double[] p : testPoints) {
            double[][] J = computeJacobian(phi, p, eps);
            double norm = 0;

            for (int i = 0; i < n; i++) {
                double rowSum = 0;
                for (int j = 0; j < n; j++) {
                    rowSum += Math.abs(J[i][j]);
                }
                norm = Math.max(norm, rowSum);
            }

            if (norm >= 1.0) {
                System.out.printf("Норма Якоби = %.4f в точке %s\n", norm, Arrays.toString(p));
                return false;
            }
        }

        return true;
    }

    // Генерация тестовых точек в окрестности
    private static double[][] generateTestPoints(double[] center, double radius) {
        int n = center.length;
        int numPoints = 1 << n; // 2^n точек (все комбинации +-radius)
        double[][] points = new double[numPoints][n];

        for (int i = 0; i < numPoints; i++) {
            for (int j = 0; j < n; j++) {
                points[i][j] = center[j] + (((i >> j) & 1) == 1 ? radius : -radius);
            }
        }

        return points;
    }

    public static double determinant(double[][] A) {
        int n = A.length;
        if (n == 2) {
            return A[0][0] * A[1][1] - A[0][1] * A[1][0];
        } else if (n == 3) {
            return A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                    - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                    + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
        } else {
            throw new UnsupportedOperationException("n > 3");
        }
    }

    public static double[] newton(
            Function<double[], double[]> F,
            double[] initialGuess,
            int maxIterations,
            double eps) {

        int n = initialGuess.length;
        double[] x = Arrays.copyOf(initialGuess, n);

        double[][] J = computeJacobian(F, x, eps);
        if (Math.abs(determinant(J)) < eps) {
            System.out.println("Ошибка: Якобиан вырожден в начальной точке.");
            return null;
        }


        for (int iter = 0; iter < maxIterations; iter++) {
            double[] Fx = F.apply(x);
            J = computeJacobian(F, x, eps); // Численный Якобиан

            // Решаем линейную систему J * deltaX = -Fx
            double[] deltaX = solveLinearSystem(J, Fx);

            for (int i = 0; i < n; i++) {
                x[i] += deltaX[i];
            }

            // Проверка на сходимость
            double error = 0;
            for (double d : deltaX) {
                error = Math.max(error, Math.abs(d));
            }

            if (error < eps) {
                System.out.printf("Сходимость за %d итераций.\n", iter + 1);
                return x;
            }
        }

        System.out.println("Достигнут максимум итераций.");
        return null;
    }

    // Численное вычисление Якобиана
    public static double[][] computeJacobian(
            Function<double[], double[]> F,
            double[] x,
            double eps) {

        int n = x.length;
        double[][] J = new double[n][n];
        double[] Fx = F.apply(x);

        for (int j = 0; j < n; j++) {
            double[] xPlusH = Arrays.copyOf(x, n);
            xPlusH[j] += eps;
            double[] FxPlusH = F.apply(xPlusH);

            for (int i = 0; i < n; i++) {
                J[i][j] = (FxPlusH[i] - Fx[i]) / eps;
            }
        }

        return J;
    }

    public static double[] solveLinearSystem(double[][] J, double[] Fx) {
        int n = Fx.length;
        double[][] A = new double[n][n + 1];

        for (int i = 0; i < n; i++) {
            System.arraycopy(J[i], 0, A[i], 0, n);
            A[i][n] = -Fx[i];
        }

        for (int k = 0; k < n; k++) {
            int maxRow = k;
            for (int i = k + 1; i < n; i++) {
                if (Math.abs(A[i][k]) > Math.abs(A[maxRow][k])) {
                    maxRow = i;
                }
            }

            double[] temp = A[k];
            A[k] = A[maxRow];
            A[maxRow] = temp;

            for (int i = k + 1; i < n; i++) {
                double factor = A[i][k] / A[k][k];
                for (int j = k; j <= n; j++) {
                    A[i][j] -= factor * A[k][j];
                }
            }
        }

        double[] deltaX = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0;
            for (int j = i + 1; j < n; j++) {
                sum += A[i][j] * deltaX[j];
            }
            deltaX[i] = (A[i][n] - sum) / A[i][i];
        }

        return deltaX;
    }
}
