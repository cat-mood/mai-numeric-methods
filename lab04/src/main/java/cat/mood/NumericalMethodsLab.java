package cat.mood;

import java.util.Arrays;

public class NumericalMethodsLab {

    // Функции, определяющие систему ОДУ
    interface ODEFunction {
        double f(double x, double y, double z);
        double g(double x, double y, double z);
    }

    // Точное решение для сравнения
    interface ExactSolution {
        double y(double x);
        double z(double x);
    }

    // Решение методом Эйлера
    public static double[][] eulerMethod(ODEFunction ode, ExactSolution exact,
                                         double x0, double y0, double z0,
                                         double h, int steps) {
        double[][] result = new double[steps + 1][5]; // x, y, z, y_exact, error
        result[0][0] = x0;
        result[0][1] = y0;
        result[0][2] = z0;
        result[0][3] = exact.y(x0);
        result[0][4] = 0.0;

        for (int i = 1; i <= steps; i++) {
            double x = result[i-1][0];
            double y = result[i-1][1];
            double z = result[i-1][2];

            double dy = h * ode.f(x, y, z);
            double dz = h * ode.g(x, y, z);

            result[i][0] = x + h;
            result[i][1] = y + dy;
            result[i][2] = z + dz;
            result[i][3] = exact.y(result[i][0]);
            result[i][4] = Math.abs(result[i][1] - result[i][3]);
        }

        return result;
    }

    // Решение методом Рунге-Кутты 4-го порядка
    public static double[][] rungeKutta4(ODEFunction ode, ExactSolution exact,
                                         double x0, double y0, double z0,
                                         double h, int steps) {
        double[][] result = new double[steps + 1][5]; // x, y, z, y_exact, error
        result[0][0] = x0;
        result[0][1] = y0;
        result[0][2] = z0;
        result[0][3] = exact.y(x0);
        result[0][4] = 0.0;

        for (int i = 1; i <= steps; i++) {
            double x = result[i-1][0];
            double y = result[i-1][1];
            double z = result[i-1][2];

            // Коэффициенты для y
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
            result[i][3] = exact.y(result[i][0]);
            result[i][4] = Math.abs(result[i][1] - result[i][3]);
        }

        return result;
    }

    // Решение методом Адамса 4-го порядка
    public static double[][] adams4(ODEFunction ode, ExactSolution exact,
                                    double x0, double y0, double z0,
                                    double h, int steps) {
        // Для Адамса нужны 4 начальные точки, используем Рунге-Кутта
        if (steps < 3) {
            throw new IllegalArgumentException("Для метода Адамса нужно минимум 4 точки (steps >= 3)");
        }

        double[][] result = new double[steps + 1][5]; // x, y, z, y_exact, error
        result[0][0] = x0;
        result[0][1] = y0;
        result[0][2] = z0;
        result[0][3] = exact.y(x0);
        result[0][4] = 0.0;

        // Первые 3 точки получаем методом Рунге-Кутта
        double[][] rkStart = rungeKutta4(ode, exact, x0, y0, z0, h, 3);
        for (int i = 1; i <= 3; i++) {
            System.arraycopy(rkStart[i], 0, result[i], 0, 5);
        }

        // Массивы для хранения предыдущих значений производных
        double[] fPrev = new double[4];
        double[] gPrev = new double[4];

        // Заполняем предыдущие значения производных
        for (int i = 0; i <= 3; i++) {
            fPrev[i] = ode.f(result[i][0], result[i][1], result[i][2]);
            gPrev[i] = ode.g(result[i][0], result[i][1], result[i][2]);
        }

        // Основной цикл метода Адамса
        for (int i = 4; i <= steps; i++) {
            // Вычисляем новые значения y и z
            result[i][1] = result[i-1][1] + h*(55*fPrev[3] - 59*fPrev[2] + 37*fPrev[1] - 9*fPrev[0])/24;
            result[i][2] = result[i-1][2] + h*(55*gPrev[3] - 59*gPrev[2] + 37*gPrev[1] - 9*gPrev[0])/24;
            result[i][0] = result[i-1][0] + h;
            result[i][3] = exact.y(result[i][0]);

            // Обновляем массив предыдущих значений производных
            System.arraycopy(fPrev, 1, fPrev, 0, 3);
            System.arraycopy(gPrev, 1, gPrev, 0, 3);

            fPrev[3] = ode.f(result[i][0], result[i][1], result[i][2]);
            gPrev[3] = ode.g(result[i][0], result[i][1], result[i][2]);

            result[i][4] = Math.abs(result[i][1] - result[i][3]);
        }

        return result;
    }

    // Метод Рунге-Ромберга для оценки погрешности
    public static double rungeRombergError(double[][] solutionH, double[][] solutionH2, int p) {
        int n = solutionH.length - 1;
        double error = 0.0;

        for (int i = 0; i <= n; i++) {
            double yH = solutionH[i][1];
            double yH2 = solutionH2[2*i][1];
            double currentError = Math.abs(yH - yH2) / (Math.pow(2, p) - 1);
            if (currentError > error) {
                error = currentError;
            }
        }

        return error;
    }

    public static void main(String[] args) {
        // Уравнение:
        // y'' + 1/x * y' + 2/x * y = 0
        // Преобразуем в систему:
        // y' = z
        // z' = -1/x * z - 2/x * y
        ODEFunction ode = new ODEFunction() {
            @Override
            public double f(double x, double y, double z) {
                return z;
            }

            @Override
            public double g(double x, double y, double z) {
                return -1.0/x * z - 2.0/x * y;
            }
        };

        // Точное решение:
        // y(x) = (cos2 - sin2) * cos(2x^(1/2)) + (cos2 + sin2) * sin(2x^(1/2))
        ExactSolution exact = new ExactSolution() {
            @Override
            public double y(double x) {
                return (Math.cos(2) - Math.sin(2)) * Math.cos(2 * Math.sqrt(x))
                        + (Math.cos(2) + Math.sin(2)) * Math.sin(2 * Math.sqrt(x));
            }

            @Override
            public double z(double x) {
                return (Math.cos(2 * Math.sqrt(x)) * (Math.cos(2) + Math.sin(2))
                        + Math.sin(2 * Math.sqrt(x)) * (Math.sin(2) - Math.cos(2))) / Math.sqrt(x);
            }
        };

        // Начальные условия
        double x0 = 1.0;
        double y0 = 1.0;
        double z0 = 1.0;
        double h = 0.025;
        int steps = 40;


        // Решение разными методами
        System.out.println("Метод Эйлера:");
        double[][] eulerSolution = eulerMethod(ode, exact, x0, y0, z0, h, steps);
        printSolution(eulerSolution);
//        GraphPlotter.plotSolutions(eulerSolution, "Метод Эйлера");

        System.out.println("\nМетод Рунге-Кутты 4-го порядка:");
        double[][] rk4Solution = rungeKutta4(ode, exact, x0, y0, z0, h, steps);
        printSolution(rk4Solution);
//        GraphPlotter.plotSolutions(eulerSolution, "Метод Рунге-Кутты 4-го порядка");

        System.out.println("\nМетод Адамса 4-го порядка:");
        double[][] adamsSolution = adams4(ode, exact, x0, y0, z0, h, steps);
        printSolution(adamsSolution);
//        GraphPlotter.plotSolutions(eulerSolution, "Метод Адамса 4-го порядка");

        GraphPlotter.plotAllSolutions(eulerSolution, rk4Solution, adamsSolution);

        // Оценка погрешности методом Рунге-Ромберга
        double h2 = h / 2;
        int steps2 = steps * 2;

        double[][] rk4SolutionH = rungeKutta4(ode, exact, x0, y0, z0, h, steps);
        double[][] rk4SolutionH2 = rungeKutta4(ode, exact, x0, y0, z0, h2, steps2);

//        double rrError = rungeRombergError(rk4SolutionH, rk4SolutionH2, 4);
//        System.out.printf("\nОценка погрешности методом Рунге-Ромберга: %.8f\n", rrError);
    }

    // Вспомогательная функция для вывода результатов
    private static void printSolution(double[][] solution) {
        System.out.println("x\t\ty числ.\t\ty точн.\t\tПогрешность\tz числ.");
        for (double[] row : solution) {
            System.out.printf("%.4f\t%.8f\t%.8f\t%.8f\t%.8f\n",
                    row[0], row[1], row[3], row[4], row[2]);
        }
    }
}