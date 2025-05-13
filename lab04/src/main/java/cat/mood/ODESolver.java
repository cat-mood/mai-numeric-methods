package cat.mood;

import java.util.ArrayList;
import java.util.List;

public class ODESolver {

    // Функция f(x, y, y') для уравнения y'' = f(x, y, y')
    public interface SecondOrderODE {
        double evaluate(double x, double y, double dy);
    }

    // Точное решение y(x)
    public interface ExactSolution {
        double evaluate(double x);
    }

    // Метод Эйлера для ОДУ 2-го порядка
    public static List<double[]> eulerMethod(SecondOrderODE ode, ExactSolution exact,
                                             double x0, double y0, double dy0,
                                             double h, double xEnd) {
        List<double[]> results = new ArrayList<>();
        double x = x0;
        double y = y0;
        double dy = dy0;

        while (x <= xEnd + 1e-9) {
            double exactY = exact.evaluate(x);
            double error = Math.abs(y - exactY);
            results.add(new double[]{x, y, dy, exactY, error});

            double d2y = ode.evaluate(x, y, dy);
            y += h * dy;
            dy += h * d2y;
            x += h;
        }

        return results;
    }

    // Метод Рунге-Кутты 4-го порядка для ОДУ 2-го порядка
    public static List<double[]> rungeKutta4Method(SecondOrderODE ode, ExactSolution exact,
                                                   double x0, double y0, double dy0,
                                                   double h, double xEnd) {
        List<double[]> results = new ArrayList<>();
        double x = x0;
        double y = y0;
        double dy = dy0;

        while (x <= xEnd + 1e-9) {
            double exactY = exact.evaluate(x);
            double error = Math.abs(y - exactY);
            results.add(new double[]{x, y, dy, exactY, error});

            // Шаг 1
            double k1y = dy;
            double k1dy = ode.evaluate(x, y, dy);

            // Шаг 2
            double k2y = dy + h * k1dy / 2;
            double k2dy = ode.evaluate(x + h/2, y + h * k1y / 2, dy + h * k1dy / 2);

            // Шаг 3
            double k3y = dy + h * k2dy / 2;
            double k3dy = ode.evaluate(x + h/2, y + h * k2y / 2, dy + h * k2dy / 2);

            // Шаг 4
            double k4y = dy + h * k3dy;
            double k4dy = ode.evaluate(x + h, y + h * k3y, dy + h * k3dy);

            // Обновление
            y += h * (k1y + 2*k2y + 2*k3y + k4y) / 6;
            dy += h * (k1dy + 2*k2dy + 2*k3dy + k4dy) / 6;
            x += h;
        }

        return results;
    }

    // Метод Адамса 4-го порядка для ОДУ 2-го порядка (использует Рунге-Кутты для стартовых точек)
    public static List<double[]> adams4Method(SecondOrderODE ode, ExactSolution exact,
                                              double x0, double y0, double dy0,
                                              double h, double xEnd) {
        List<double[]> results = new ArrayList<>();

        // Используем Рунге-Кутты для получения первых 4 точек
        List<double[]> rk4Results = rungeKutta4Method(ode, exact, x0, y0, dy0, h, x0 + 3*h);
        for (int i = 0; i < 4 && i < rk4Results.size(); i++) {
            results.add(rk4Results.get(i));
        }

        if (results.size() < 4) {
            return results;
        }

        // Подготовка истории для Адамса
        double[] x = new double[4];
        double[] y = new double[4];
        double[] dy = new double[4];
        double[] f = new double[4];

        for (int i = 0; i < 4; i++) {
            x[i] = results.get(i)[0];
            y[i] = results.get(i)[1];
            dy[i] = results.get(i)[2];
            f[i] = ode.evaluate(x[i], y[i], dy[i]);
        }

        // Основной цикл Адамса
        for (int i = 3; x[i] < xEnd - 1e-9; i++) {
            double xNext = x[i] + h;

            // Прогноз
            double yPred = y[i] + h * (55*dy[i] - 59*dy[i-1] + 37*dy[i-2] - 9*dy[i-3]) / 24;
            double dyPred = dy[i] + h * (55*f[i] - 59*f[i-1] + 37*f[i-2] - 9*f[i-3]) / 24;
            double fPred = ode.evaluate(xNext, yPred, dyPred);

            // Коррекция
            double yNext = y[i] + h * (9*dyPred + 19*dy[i] - 5*dy[i-1] + dy[i-2]) / 24;
            double dyNext = dy[i] + h * (9*fPred + 19*f[i] - 5*f[i-1] + f[i-2]) / 24;
            double fNext = ode.evaluate(xNext, yNext, dyNext);

            // Обновление массивов
            for (int j = 0; j < 3; j++) {
                x[j] = x[j+1];
                y[j] = y[j+1];
                dy[j] = dy[j+1];
                f[j] = f[j+1];
            }
            x[3] = xNext;
            y[3] = yNext;
            dy[3] = dyNext;
            f[3] = fNext;

            // Сохранение результата
            double exactY = exact.evaluate(xNext);
            double error = Math.abs(yNext - exactY);
            results.add(new double[]{xNext, yNext, dyNext, exactY, error});
        }

        return results;
    }

    // Метод Рунге-Ромберга для оценки погрешности
    public static double rungeRombergError(List<double[]> fineResults, List<double[]> coarseResults) {
        if (fineResults.isEmpty() || coarseResults.isEmpty()) return 0;

        int n = fineResults.size();
        int m = coarseResults.size();
        int step = n / m;
        if (step <= 0) return 0;

        double error = 0;
        int count = 0;

        for (int i = 0; i < m && i*step < n; i++) {
            double fineY = fineResults.get(i*step)[1];
            double coarseY = coarseResults.get(i)[1];
            error += Math.abs(fineY - coarseY);
            count++;
        }

        return error / count;
    }

    public static void main(String[] args) {
        // Пример: y'' + y = 0, y(0) = 0, y'(0) = 1
        // Точное решение: y(x) = sin(x)
        SecondOrderODE ode = (x, y, dy) -> - 1 / x * dy - 2 / x * y;
        ExactSolution exact = x -> (Math.cos(2) - Math.sin(2)) * Math.cos(2 * Math.sqrt(x))
                + (Math.cos(2) + Math.sin(2)) * Math.sin(2 * Math.sqrt(x));

        double x0 = 1;
        double y0 = 1;
        double dy0 = 1;
        double h = 0.1;
        double xEnd = 2;

        System.out.println("Метод Эйлера:");
        List<double[]> eulerResults = eulerMethod(ode, exact, x0, y0, dy0, h, xEnd);
        printResults(eulerResults);

        System.out.println("\nМетод Рунге-Кутты 4-го порядка:");
        List<double[]> rk4Results = rungeKutta4Method(ode, exact, x0, y0, dy0, h, xEnd);
        printResults(rk4Results);

        System.out.println("\nМетод Адамса 4-го порядка:");
        List<double[]> adamsResults = adams4Method(ode, exact, x0, y0, dy0, h, xEnd);
        printResults(adamsResults);

        // Оценка погрешности методом Рунге-Ромберга
        List<double[]> rk4FineResults = rungeKutta4Method(ode, exact, x0, y0, dy0, h/2, xEnd);
        double rrError = rungeRombergError(rk4FineResults, rk4Results);
        System.out.printf("\nОценка погрешности методом Рунге-Ромберга: %.6f\n", rrError);
    }

    private static void printResults(List<double[]> results) {
        System.out.println("x\t\tЧисл. y(x)\tТочн. y(x)\tПогрешность");
        for (double[] row : results) {
            System.out.printf("%.4f\t%.6f\t%.6f\t%.6f\n",
                    row[0], row[1], row[3], row[4]);
        }
    }
}
