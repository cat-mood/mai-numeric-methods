package cat.mood;

public class Derivative {
    public static void main(String[] args) {
        double[] x = {-1, 0, 1, 2, 3};
        double[] y = {-1.7854, 0, 1.7854, 3.1071, 4.249};

        double xStar = 1.0;

        try {
            // Первая производная
            double firstDerivLeft = firstDerivativeNewton1Left(x, y, xStar);
            double firstDerivRight = firstDerivativeNewton1Right(x, y, xStar);
            double firstDerivSecondOrder = firstDerivativeNewton2(x, y, xStar);

            // Вторая производная
            double secondDeriv = secondDerivativeNewton2(x, y, xStar);

            System.out.println("Первая производная (левосторонняя, 1-й порядок): " + firstDerivLeft);
            System.out.println("Первая производная (правосторонняя, 1-й порядок): " + firstDerivRight);
            System.out.println("Первая производная (2-й порядок точности): " + firstDerivSecondOrder);
            System.out.println("Вторая производная (2-й порядок точности): " + secondDeriv);

        } catch (IllegalArgumentException e) {
            System.out.println("Ошибка: " + e.getMessage());
        }
    }

    // Вычисление разделенных разностей для полинома Ньютона
    private static double[][] getDividedDifferences(double[] x, double[] y) {
        int n = x.length;
        double[][] f = new double[n][n];

        for (int i = 0; i < n; i++) {
            f[i][0] = y[i];
        }

        for (int j = 1; j < n; j++) {
            for (int i = 0; i < n - j; i++) {
                f[i][j] = (f[i+1][j-1] - f[i][j-1]) / (x[i+j] - x[i]);
            }
        }

        return f;
    }

    // Первая производная через полином Ньютона 1-й степени (левосторонняя)
    public static double firstDerivativeNewton1Left(double[] x, double[] y, double xStar) {
        validateInput(x, y, xStar);
        int index = findIndex(x, xStar);

        if (index == 0) {
            throw new IllegalArgumentException("Недостаточно точек для левосторонней производной");
        }

        double h = x[index] - x[index-1];
        return (y[index] - y[index-1]) / h;
    }

    // Первая производная через полином Ньютона 1-й степени (правосторонняя)
    public static double firstDerivativeNewton1Right(double[] x, double[] y, double xStar) {
        validateInput(x, y, xStar);
        int index = findIndex(x, xStar);

        if (index == x.length - 1) {
            throw new IllegalArgumentException("Недостаточно точек для правосторонней производной");
        }

        double h = x[index+1] - x[index];
        return (y[index+1] - y[index]) / h;
    }

    // Первая производная через полином Ньютона 2-й степени
    public static double firstDerivativeNewton2(double[] x, double[] y, double xStar) {
        validateInput(x, y, xStar);
        int index = findIndex(x, xStar);

        if (index == 0 || index == x.length - 1) {
            throw new IllegalArgumentException("Для метода 2-го порядка нужны точки по обе стороны");
        }

        double[][] f = getDividedDifferences(x, y);

        // Производная полинома P(x) = f[x0] + f[x0,x1](x-x0) + f[x0,x1,x2](x-x0)(x-x1)
        // P'(x) = f[x0,x1] + f[x0,x1,x2]*(2x - x0 - x1)

        double x0 = x[index-1];
        double x1 = x[index];
        double x2 = x[index+1];

        double f01 = f[index-1][1];
        double f012 = f[index-1][2];

        return f01 + f012 * (2*xStar - x0 - x1);
    }

    // Вторая производная через полином Ньютона 2-й степени
    public static double secondDerivativeNewton2(double[] x, double[] y, double xStar) {
        validateInput(x, y, xStar);
        int index = findIndex(x, xStar);

        if (index == 0 || index == x.length - 1) {
            throw new IllegalArgumentException("Для второй производной нужны точки по обе стороны");
        }

        double[][] f = getDividedDifferences(x, y);

        // Вторая производная полинома 2-й степени:
        // P''(x) = 2*f[x0,x1,x2]

        return 2 * f[index-1][2];
    }

    private static int findIndex(double[] x, double xStar) {
        for (int i = 0; i < x.length; i++) {
            if (Math.abs(x[i] - xStar) < 1e-9) {
                return i;
            }
        }
        throw new IllegalArgumentException("Точка X* не найдена в массиве x");
    }

    private static void validateInput(double[] x, double[] y, double xStar) {
        if (x == null || y == null || x.length != y.length || x.length < 2) {
            throw new IllegalArgumentException("Некорректные входные данные");
        }
    }
}
