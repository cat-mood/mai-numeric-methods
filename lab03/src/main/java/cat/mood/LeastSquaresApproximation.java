package cat.mood;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.awt.*;
import java.awt.geom.Ellipse2D;

public class LeastSquaresApproximation {

    public static void main(String[] args) {
        // Пример входных данных
        double[] x = {-0.7, -0.4, -0.1, 0.2, 0.5, 0.8};
        double[] y = {-1.4754, -0.81152, -0.20017, 0.40136, 1.0236, 1.7273};

        // Приближение многочленом 1-ой степени
        double[] linearCoeffs = leastSquares(x, y, 1);
        System.out.println("Многочлен 1-ой степени: y = " + linearCoeffs[0] + " + " + linearCoeffs[1] + "x");
        double linearError = calculateError(x, y, linearCoeffs);
        System.out.println("Сумма квадратов ошибок (1-ая степень): " + linearError);

        // Приближение многочленом 2-ой степени
        double[] quadraticCoeffs = leastSquares(x, y, 2);
        System.out.println("Многочлен 2-ой степени: y = " + quadraticCoeffs[0] + " + " + quadraticCoeffs[1] + "x + " + quadraticCoeffs[2] + "x^2");
        double quadraticError = calculateError(x, y, quadraticCoeffs);
        System.out.println("Сумма квадратов ошибок (2-ая степень): " + quadraticError);

        // Построение графиков
        plotFunctionAndApproximations(x, y, linearCoeffs, quadraticCoeffs);
    }

    // Метод наименьших квадратов для нахождения коэффициентов многочлена степени n
    public static double[] leastSquares(double[] x, double[] y, int n) {
        int m = x.length;
        double[][] A = new double[n + 1][n + 1];
        double[] B = new double[n + 1];

        // Заполнение матрицы A и вектора B
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= n; j++) {
                for (int k = 0; k < m; k++) {
                    A[i][j] += Math.pow(x[k], i + j);
                }
            }
            for (int k = 0; k < m; k++) {
                B[i] += y[k] * Math.pow(x[k], i);
            }
        }

        // Решение системы линейных уравнений методом Гаусса
        return gauss(A, B);
    }

    // Решение системы линейных уравнений методом Гаусса
    public static double[] gauss(double[][] A, double[] B) {
        int n = B.length;
        for (int p = 0; p < n; p++) {
            // Поиск максимального элемента в текущем столбце
            int max = p;
            for (int i = p + 1; i < n; i++) {
                if (Math.abs(A[i][p]) > Math.abs(A[max][p])) {
                    max = i;
                }
            }
            // Обмен строками
            double[] temp = A[p];
            A[p] = A[max];
            A[max] = temp;
            double t = B[p];
            B[p] = B[max];
            B[max] = t;

            // Приведение к треугольному виду
            for (int i = p + 1; i < n; i++) {
                double alpha = A[i][p] / A[p][p];
                B[i] -= alpha * B[p];
                for (int j = p; j < n; j++) {
                    A[i][j] -= alpha * A[p][j];
                }
            }
        }

        // Обратный ход
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < n; j++) {
                sum += A[i][j] * x[j];
            }
            x[i] = (B[i] - sum) / A[i][i];
        }
        return x;
    }

    // Вычисление суммы квадратов ошибок
    public static double calculateError(double[] x, double[] y, double[] coeffs) {
        double error = 0.0;
        for (int i = 0; i < x.length; i++) {
            double approxY = 0.0;
            for (int j = 0; j < coeffs.length; j++) {
                approxY += coeffs[j] * Math.pow(x[i], j);
            }
            error += Math.pow(y[i] - approxY, 2);
        }
        return error;
    }

    // Построение графиков
    public static void plotFunctionAndApproximations(double[] x, double[] y, double[] linearCoeffs, double[] quadraticCoeffs) {
        XYSeries originalSeries = new XYSeries("Исходная функция");
        for (int i = 0; i < x.length; i++) {
            originalSeries.add(x[i], y[i]);
        }

        XYSeries linearSeries = new XYSeries("Линейная аппроксимация (красный)");
        XYSeries quadraticSeries = new XYSeries("Квадратичная аппроксимация (синий)");

        double minX = x[0];
        double maxX = x[x.length - 1];
        for (double xi = minX; xi <= maxX; xi += 0.1) {
            double linearY = linearCoeffs[0] + linearCoeffs[1] * xi;
            linearSeries.add(xi, linearY);

            double quadraticY = quadraticCoeffs[0] + quadraticCoeffs[1] * xi + quadraticCoeffs[2] * xi * xi;
            quadraticSeries.add(xi, quadraticY);
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(originalSeries);
        dataset.addSeries(linearSeries);
        dataset.addSeries(quadraticSeries);

        JFreeChart chart = ChartFactory.createXYLineChart(
                "Аппроксимация методом наименьших квадратов",
                "X",
                "Y",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        XYPlot plot = chart.getXYPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();

        // Настройки для исходных данных (чёрные точки)
        renderer.setSeriesPaint(0, Color.BLACK);
        renderer.setSeriesLinesVisible(0, false);
        renderer.setSeriesShapesVisible(0, true);
        renderer.setSeriesShape(0, new Ellipse2D.Double(-3, -3, 6, 6)); // Круглые точки

        // Настройки для линейной аппроксимации (красная линия)
        renderer.setSeriesPaint(1, Color.RED);
        renderer.setSeriesLinesVisible(1, true);
        renderer.setSeriesShapesVisible(1, false);
        renderer.setSeriesStroke(1, new BasicStroke(2.0f)); // Толщина линии

        // Настройки для квадратичной аппроксимации (синяя линия)
        renderer.setSeriesPaint(2, Color.BLUE);
        renderer.setSeriesLinesVisible(2, true);
        renderer.setSeriesShapesVisible(2, false);
        renderer.setSeriesStroke(2, new BasicStroke(2.0f));

        plot.setRenderer(renderer);

        ChartFrame frame = new ChartFrame("Графики аппроксимации", chart);
        frame.pack();
        frame.setVisible(true);
    }
}
