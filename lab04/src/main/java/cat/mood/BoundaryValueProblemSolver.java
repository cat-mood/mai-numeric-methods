package cat.mood;

import java.util.Arrays;

public class BoundaryValueProblemSolver {

    interface ODEFunction {
        double f(double x, double y, double z);
        double g(double x, double y, double z);
    }

    interface ExactSolution {
        double y(double x);
        double z(double x);
    }

    // Метод стрельбы для смешанных граничных условий
    public static double[][] shootingMethod(ODEFunction ode, ExactSolution exact,
                                            double a, double b, double alpha, double beta, double gamma,
                                            double h, double eps) {
        // Начальные предположения для y(a)
        double eta0 = 1.0;
        double eta1 = 0.5;

        // Решаем задачу Коши с начальными условиями y'(a)=alpha, y(a)=eta0
        double[][] sol0 = rungeKutta4System(ode, a, eta0, alpha, h, (int)((b-a)/h));
        double phi0 = beta*sol0[sol0.length-1][1] + gamma*sol0[sol0.length-1][2] - 23*Math.exp(-4);

        // Решаем задачу Коши с начальными условиями y'(a)=alpha, y(a)=eta1
        double[][] sol1 = rungeKutta4System(ode, a, eta1, alpha, h, (int)((b-a)/h));
        double phi1 = beta*sol1[sol1.length-1][1] + gamma*sol1[sol1.length-1][2] - 23*Math.exp(-4);

        // Метод секущих для нахождения правильного eta
        double eta = eta1;
        double phi = phi1;
        int iterations = 0;

        while (Math.abs(phi) > eps && iterations < 100) {
            double etaNew = eta1 - (eta1 - eta0)/(phi1 - phi0)*phi1;

            eta0 = eta1;
            eta1 = etaNew;
            phi0 = phi1;

            double[][] solNew = rungeKutta4System(ode, a, eta1, alpha, h, (int)((b-a)/h));
            phi1 = beta*solNew[solNew.length-1][1] + gamma*solNew[solNew.length-1][2] - 23*Math.exp(-4);

            eta = eta1;
            phi = phi1;
            iterations++;
        }

        // Финальное решение с найденным eta
        double[][] finalSolution = rungeKutta4System(ode, a, eta, alpha, h, (int)((b-a)/h));

        // Добавляем точные значения и погрешности
        double[][] result = new double[finalSolution.length][5];
        for (int i = 0; i < finalSolution.length; i++) {
            result[i][0] = finalSolution[i][0]; // x
            result[i][1] = finalSolution[i][1]; // y
            result[i][2] = finalSolution[i][2]; // z
            result[i][3] = exact.y(finalSolution[i][0]); // y_exact
            result[i][4] = Math.abs(result[i][1] - result[i][3]); // error
        }

        return result;
    }

    // Исправленный конечно-разностный метод
    public static double[][] finiteDifferenceMethod(ODEFunction ode, ExactSolution exact,
                                                    double a, double b, double alpha, double beta, double gamma,
                                                    double h) {
        int n = (int)((b - a)/h);
        double[] x = new double[n+1];
        for (int i = 0; i <= n; i++) {
            x[i] = a + i*h;
        }

        // Коэффициенты для трехдиагональной системы
        double[] A = new double[n+1];
        double[] B = new double[n+1];
        double[] C = new double[n+1];
        double[] D = new double[n+1];

        // Левое граничное условие y'(0) = 1 (используем одностороннюю разность)
        A[0] = 0;
        B[0] = -1.0;
        C[0] = 1.0;
        D[0] = h*alpha;

        // Правое граничное условие 4y(2) - y'(2) = 23e^-4 (используем одностороннюю разность)
        A[n] = -1.0;
        B[n] = beta*h + gamma;
        C[n] = 0;
        D[n] = h*23*Math.exp(-4);

        // Заполняем коэффициенты для внутренних точек
        for (int i = 1; i < n; i++) {
            double xi = x[i];
            A[i] = 1.0 - h*2*xi; // Коэффициент при y_{i-1}
            B[i] = -2.0 + h*h*(4*xi*xi + 2); // Коэффициент при y_i
            C[i] = 1.0 + h*2*xi; // Коэффициент при y_{i+1}
            D[i] = 0; // Правая часть
        }

        // Решаем трехдиагональную систему методом прогонки
        double[] y = solveTridiagonalSystem(A, B, C, D);

        // Создаем результат с вычисленными значениями z по конечно-разностной схеме
        double[][] result = new double[n+1][5];
        for (int i = 0; i <= n; i++) {
            result[i][0] = x[i];
            result[i][1] = y[i];

            // Вычисляем z = y' с помощью конечных разностей
            if (i == 0) {
                result[i][2] = alpha; // используем граничное условие
            } else if (i == n) {
                result[i][2] = 4*y[i] - 23*Math.exp(-4); // из правого граничного условия
            } else {
                result[i][2] = (y[i+1] - y[i-1])/(2*h); // центральная разность
            }

            result[i][3] = exact.y(x[i]);
            result[i][4] = Math.abs(y[i] - result[i][3]);
        }

        return result;
    }

    // Исправленный метод прогонки
    private static double[] solveTridiagonalSystem(double[] A, double[] B, double[] C, double[] D) {
        int n = B.length - 1;
        double[] alpha = new double[n+1];
        double[] beta = new double[n+1];

        // Прямой ход
        alpha[1] = C[0]/B[0];
        beta[1] = D[0]/B[0];

        for (int i = 1; i < n; i++) {
            double denominator = B[i] - A[i]*alpha[i];
            alpha[i+1] = C[i] / denominator;
            beta[i+1] = (D[i] - A[i]*beta[i]) / denominator;
        }

        // Обратный ход
        double[] y = new double[n+1];
        y[n] = (D[n] - A[n]*beta[n]) / (B[n] - A[n]*alpha[n]);

        for (int i = n-1; i >= 0; i--) {
            y[i] = alpha[i+1]*y[i+1] + beta[i+1];
        }

        return y;
    }

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

    public static double rungeRombergError(double[][] solutionH, double[][] solutionH2, int p) {
        double maxError = 0.0;
        int n = solutionH.length;

        for (int i = 0; i < n; i++) {
            double yH = solutionH[i][1];
            double yH2 = solutionH2[2*i][1];
            double error = Math.abs(yH - yH2) / (Math.pow(2, p) - 1);
            if (error > maxError) {
                maxError = error;
            }
        }

        return maxError;
    }

    public static void main(String[] args) {
        // Уравнение: y'' + 4xy' + (4x^2 + 2)y = 0
        ODEFunction ode = new ODEFunction() {
            @Override
            public double f(double x, double y, double z) {
                return z;
            }

            @Override
            public double g(double x, double y, double z) {
                return -4*x*z - (4*x*x + 2)*y;
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
        double b = 2.0;
        double alpha = 1.0; // y'(0) = 1
        double beta = 4.0;  // 4y(2) - y'(2) = 23e^-4
        double gamma = -1.0;
        double h = 0.1;
        double eps = 1e-6;

        // Решение методом стрельбы
        System.out.println("Метод стрельбы:");
        double[][] shootingSolution = shootingMethod(ode, exact, a, b, alpha, beta, gamma, h, eps);
        printSolution(shootingSolution);

        // Решение конечно-разностным методом
        System.out.println("\nКонечно-разностный метод:");
        double[][] fdSolution = finiteDifferenceMethod(ode, exact, a, b, alpha, beta, gamma, h);
        printSolution(fdSolution);

        // Оценка погрешности методом Рунге-Ромберга для метода стрельбы
        double h2 = h/2;
        double[][] shootingSolutionH = shootingMethod(ode, exact, a, b, alpha, beta, gamma, h, eps);
        double[][] shootingSolutionH2 = shootingMethod(ode, exact, a, b, alpha, beta, gamma, h2, eps);
        double rrErrorShooting = rungeRombergError(shootingSolutionH, shootingSolutionH2, 4);
        System.out.printf("\nОценка погрешности метода стрельбы (Рунге-Ромберг): %.8f\n", rrErrorShooting);

        // Оценка погрешности методом Рунге-Ромберга для конечно-разностного метода
        double[][] fdSolutionH = finiteDifferenceMethod(ode, exact, a, b, alpha, beta, gamma, h);
        double[][] fdSolutionH2 = finiteDifferenceMethod(ode, exact, a, b, alpha, beta, gamma, h2);
        double rrErrorFD = rungeRombergError(fdSolutionH, fdSolutionH2, 2);
        System.out.printf("Оценка погрешности конечно-разностного метода (Рунге-Ромберг): %.8f\n", rrErrorFD);
    }

    private static void printSolution(double[][] solution) {
        System.out.println("x\t\ty числ.\t\ty точн.\t\tПогрешность\tz числ.");
        for (double[] row : solution) {
            System.out.printf("%.4f\t%.8f\t%.8f\t%.8f\t%.8f\n",
                    row[0], row[1], row[3], row[4], row[2]);
        }
    }
}
