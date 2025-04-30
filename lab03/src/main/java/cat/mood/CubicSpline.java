package cat.mood;

import java.util.Arrays;

public class CubicSpline {
    private double[] x; // Узлы интерполяции
    private double[] y; // Значения функции в узлах
    private double[] a, b, c, d; // Коэффициенты сплайна

    public CubicSpline(double[] x, double[] y) {
        if (x == null || y == null || x.length != y.length || x.length < 2) {
            throw new IllegalArgumentException("Некорректные входные данные");
        }

        this.x = Arrays.copyOf(x, x.length);
        this.y = Arrays.copyOf(y, y.length);
        calculateCoefficients();
    }

    private void calculateCoefficients() {
        final int n = x.length;
        a = Arrays.copyOf(y, n);
        b = new double[n];
        d = new double[n];
        c = new double[n];

        double[] h = new double[n - 1];
        for (int i = 0; i < n - 1; i++) {
            h[i] = x[i + 1] - x[i];
        }

        double[] alpha = new double[n - 1];
        for (int i = 1; i < n - 1; i++) {
            alpha[i] = 3 * ((a[i + 1] - a[i]) / h[i] - (a[i] - a[i - 1]) / h[i - 1]);
        }

        // Метод прогонки с коэффициентами P и Q
        double[] P = new double[n];
        double[] Q = new double[n];

        // Естественные граничные условия (c[0] = 0)
        c[0] = 0;

        // Прямой ход метода прогонки
        P[1] = -h[1] / (2 * (h[0] + h[1]));
        Q[1] = alpha[1] / (2 * (h[0] + h[1]));

        for (int i = 2; i < n - 1; i++) {
            double denominator = 2 * (h[i - 1] + h[i]) + h[i - 1] * P[i - 1];
            P[i] = -h[i] / denominator;
            Q[i] = (alpha[i] - h[i - 1] * Q[i - 1]) / denominator;
        }

        c[n - 1] = Q[n - 1];
        // Обратный ход метода прогонки
        for (int i = n - 2; i >= 1; i--) {
            c[i] = P[i] * c[i + 1] + Q[i];
        }

        // Вычисляем коэффициенты b и d
        for (int i = 0; i < n - 1; i++) {
            b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3;
            d[i] = (c[i + 1] - c[i]) / (3 * h[i]);
        }

        b[n - 1] = (y[n - 1] - y[n - 2]) / h[n - 2] - ((double) 2 / 3) * h[n - 2] * c[n - 1];
        d[n - 1] = - c[n - 1] / (3 * h[n - 2]);
    }

    public double interpolate(double xValue) {
        if (xValue < x[0] || xValue > x[x.length - 1]) {
            throw new IllegalArgumentException("x вне диапазона интерполяции");
        }

        int i = 0;
        // Находим интервал, в который попадает xValue
        while (i < x.length - 1 && xValue > x[i + 1]) {
            i++;
        }

        double dx = xValue - x[i];
        return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
    }

    public static void main(String[] args) {
        double[] x = {-0.4, -0.1, 0.2, 0.5, 0.8};
        double[] y = {-0.81152, -0.20017, 0.40136, 1.0236, 1.7273};

        int n = x.length;

        CubicSpline spline = new CubicSpline(x, y);

        double t = 0.1;
        double value = spline.interpolate(t);
        System.out.printf("Значение сплайна в x = %f: %f\n", t, value);

        System.out.println("Коэффициенты: ");
        System.out.println("a = " + Arrays.toString(spline.a));
        System.out.println("b = " + Arrays.toString(spline.b));
        System.out.println("c = " + Arrays.toString(spline.c));
        System.out.println("d = " + Arrays.toString(spline.d));
    }
}