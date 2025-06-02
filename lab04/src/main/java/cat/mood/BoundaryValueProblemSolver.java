package cat.mood;

import java.util.Arrays;

public class BoundaryValueProblemSolver {

    interface ODEFunction {
        double f(double x, double y, double z); // z = y'
        double g(double x, double y, double z); // z' = y''
    }

    interface ExactSolution {
        double y(double x);
        double z(double x);
    }

    // Метод стрельбы для краевых условий y(a) = ya, y(b) = yb
    public static double[][] shootingMethod(ODEFunction ode, ExactSolution exact,
                                            double a, double b, double ya, double yb,
                                            double h, double eps) {
        // Начальные предположения для y'(a)
        double eta0 = 1.0;
        double eta1 = 0.5;

        // Первое приближение
        double[][] sol0 = rungeKutta4System(ode, a, ya, eta0, h, (int)((b-a)/h));
        double phi0 = sol0[sol0.length-1][1] - yb;

        // Второе приближение
        double[][] sol1 = rungeKutta4System(ode, a, ya, eta1, h, (int)((b-a)/h));
        double phi1 = sol1[sol1.length-1][1] - yb;

        // Метод секущих для нахождения правильного eta
        double eta = eta1;
        double phi = phi1;
        int iterations = 0;

        while (Math.abs(phi) > eps && iterations < 100) {
            double etaNew = eta1 - (eta1 - eta0)/(phi1 - phi0)*phi1;

            eta0 = eta1;
            eta1 = etaNew;
            phi0 = phi1;

            double[][] solNew = rungeKutta4System(ode, a, ya, eta1, h, (int)((b-a)/h));
            phi1 = solNew[solNew.length-1][1] - yb;

            eta = eta1;
            phi = phi1;
            iterations++;
        }

        // Финальное решение
        double[][] finalSolution = rungeKutta4System(ode, a, ya, eta, h, (int)((b-a)/h));

        // Добавляем точные значения и погрешности
        double[][] result = new double[finalSolution.length][5];
        for (int i = 0; i < finalSolution.length; i++) {
            result[i][0] = finalSolution[i][0]; // x
            result[i][1] = finalSolution[i][1]; // y
            result[i][2] = finalSolution[i][2]; // z = y'
            result[i][3] = exact.y(finalSolution[i][0]); // y_exact
            result[i][4] = Math.abs(result[i][1] - result[i][3]); // error
        }

        return result;
    }

    // Конечно-разностный метод для y(a) = ya, y(b) = yb
    public static double[][] finiteDifferenceMethod(ODEFunction ode, ExactSolution exact,
                                                    double a, double b, double ya, double yb,
                                                    double h) {
        int n = (int)((b - a)/h);
        double[] x = new double[n+1];
        for (int i = 0; i <= n; i++) {
            x[i] = a + i*h;
        }

        // Коэффициенты трехдиагональной системы
        double[] A = new double[n+1];
        double[] B = new double[n+1];
        double[] C = new double[n+1];
        double[] D = new double[n+1];

        // Левое граничное условие y(0) = 1
        A[0] = 0;
        B[0] = 1;
        C[0] = 0;
        D[0] = ya;

        // Внутренние точки
        for (int i = 1; i < n; i++) {
            double xi = x[i];
            A[i] = 1 - 2*xi*h;             // y_{i-1}
            B[i] = -2 + h*h*(4*xi*xi + 2);  // y_i
            C[i] = 1 + 2*xi*h;              // y_{i+1}
            D[i] = 0;                       // правая часть
        }

        // Правое граничное условие y(1) = 2/e
        A[n] = 0;
        B[n] = 1;
        C[n] = 0;
        D[n] = yb;

        // Решаем систему
        double[] y = solveTridiagonalSystem(A, B, C, D);

        // Формируем результаты
        double[][] result = new double[n+1][5];
        for (int i = 0; i <= n; i++) {
            result[i][0] = x[i];
            result[i][1] = y[i];

            // Вычисляем производные
            if (i == 0) {
                result[i][2] = (-3*y[i] + 4*y[i+1] - y[i+2])/(2*h); // вперед
            } else if (i == n) {
                result[i][2] = (y[i-2] - 4*y[i-1] + 3*y[i])/(2*h); // назад
            } else {
                result[i][2] = (y[i+1] - y[i-1])/(2*h); // центральная
            }

            result[i][3] = exact.y(x[i]);
            result[i][4] = Math.abs(y[i] - result[i][3]);
        }

        return result;
    }

    // Метод Рунге-Кутты 4-го порядка для системы
    private static double[][] rungeKutta4System(ODEFunction ode, double x0, double y0, double z0,
                                                double h, int steps) {
        double[][] result = new double[steps+1][3];
        result[0][0] = x0;
        result[0][1] = y0;
        result[0][2] = z0;

        for (int i = 1; i <= steps; i++) {
            double x = result[i-1][0];
            double y = result[i-1][1];
            double z = result[i-1][2];

            double k1 = h * ode.f(x, y, z);
            double l1 = h * ode.g(x, y, z);

            double k2 = h * ode.f(x + h/2, y + k1/2, z + l1/2);
            double l2 = h * ode.g(x + h/2, y + k1/2, z + l1/2);

            double k3 = h * ode.f(x + h/2, y + k2/2, z + l2/2);
            double l3 = h * ode.g(x + h/2, y + k2/2, z + l2/2);

            double k4 = h * ode.f(x + h, y + k3, z + l3);
            double l4 = h * ode.g(x + h, y + k3, z + l3);

            result[i][0] = x + h;
            result[i][1] = y + (k1 + 2*k2 + 2*k3 + k4)/6;
            result[i][2] = z + (l1 + 2*l2 + 2*l3 + l4)/6;
        }

        return result;
    }

    // Метод прогонки для трехдиагональной системы
    private static double[] solveTridiagonalSystem(double[] A, double[] B, double[] C, double[] D) {
        int n = B.length - 1;
        double[] cp = new double[n+1];
        double[] dp = new double[n+1];

        // Прямой ход
        cp[0] = C[0]/B[0];
        dp[0] = D[0]/B[0];

        for (int i = 1; i <= n; i++) {
            double m = 1.0/(B[i] - A[i]*cp[i-1]);
            cp[i] = C[i]*m;
            dp[i] = (D[i] - A[i]*dp[i-1])*m;
        }

        // Обратный ход
        double[] y = new double[n+1];
        y[n] = dp[n];

        for (int i = n-1; i >= 0; i--) {
            y[i] = dp[i] - cp[i]*y[i+1];
        }

        return y;
    }

    public static void main(String[] args) {
        // Уравнение: y'' + 4xy' + (4x^2 + 2)y = 0
        ODEFunction ode = new ODEFunction() {
            @Override
            public double f(double x, double y, double z) {
                return z; // y' = z
            }

            @Override
            public double g(double x, double y, double z) {
                return -4*x*z - (4*x*x + 2)*y; // z' = y''
            }
        };

        // Точное решение: y(x) = (1 + x)e^(-x^2)
        ExactSolution exact = new ExactSolution() {
            @Override
            public double y(double x) {
                return (1 + x)*Math.exp(-x*x);
            }

            @Override
            public double z(double x) {
                return (1 - 2*x*(1 + x))*Math.exp(-x*x);
            }
        };

        // Параметры задачи
        double a = 0.0;
        double b = 1.0;
        double ya = 1.0;          // y(0) = 1
        double yb = 2.0/Math.E;   // y(1) = 2/e
        double h = 0.1;
        double eps = 1e-6;

        // Решение методом стрельбы
        System.out.println("Метод стрельбы:");
        double[][] shootingSolution = shootingMethod(ode, exact, a, b, ya, yb, h, eps);
        printSolution(shootingSolution);
        BVPGraphPlotter.plotSolutions(shootingSolution, "Метод стрельбы");

        // Решение конечно-разностным методом
        System.out.println("\nКонечно-разностный метод:");
        double[][] fdSolution = finiteDifferenceMethod(ode, exact, a, b, ya, yb, h);
        printSolution(fdSolution);
        BVPGraphPlotter.plotSolutions(fdSolution, "Конечно-разностный метод");
    }

    private static void printSolution(double[][] solution) {
        System.out.println("x\t\ty числ.\t\ty точн.\t\tПогрешность\ty' числ.");
        for (double[] row : solution) {
            System.out.printf("%.4f\t%.8f\t%.8f\t%.8f\t%.8f\n",
                    row[0], row[1], row[3], row[4], row[2]);
        }
    }
}