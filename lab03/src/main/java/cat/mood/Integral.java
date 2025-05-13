package cat.mood;

import java.util.function.Function;

public class Integral {
    public static void main(String[] args) {
        Function<Double, Double> y = x -> Math.pow(x, 2) / (625 - Math.pow(x, 4));

        double x0 = 0.0;
        double x1 = 4.0;
        double h1 = 1.0;
        double h2 = 0.5;

        // Вычисление интеграла разными методами с шагом h1
        double rectH1 = rectangleMethod(y, x0, x1, h1);
        double trapH1 = trapezoidalMethod(y, x0, x1, h1);
        double simpH1 = simpsonMethod(y, x0, x1, h1);

        // Вычисление интеграла разными методами с шагом h2
        double rectH2 = rectangleMethod(y, x0, x1, h2);
        double trapH2 = trapezoidalMethod(y, x0, x1, h2);
        double simpH2 = simpsonMethod(y, x0, x1, h2);

        // Оценка погрешности и уточнение методом Рунге-Ромберга
        double rectRefined = rungeRombergRefined(rectH1, rectH2, h1, h2, 2);
        double trapRefined = rungeRombergRefined(trapH1, trapH2, h1, h2, 2);
        double simpRefined = rungeRombergRefined(simpH1, simpH2, h1, h2, 4);

        double rectError = Math.abs(rectRefined - rectH2);
        double trapError = Math.abs(trapRefined - trapH2);
        double simpError = Math.abs(simpRefined - simpH2);

        // Вывод результатов
        System.out.println("Метод прямоугольников:");
        System.out.printf("h=%.3f: F=%.8f\n", h1, rectH1);
        System.out.printf("h=%.3f: F=%.8f\n", h2, rectH2);
        System.out.printf("Уточнённое значение: %.8f\n", rectRefined);
        System.out.printf("Погрешность: %.8f\n\n", rectError);

        System.out.println("Метод трапеций:");
        System.out.printf("h=%.3f: F=%.8f\n", h1, trapH1);
        System.out.printf("h=%.3f: F=%.8f\n", h2, trapH2);
        System.out.printf("Уточнённое значение: %.8f\n", trapRefined);
        System.out.printf("Погрешность: %.8f\n\n", trapError);

        System.out.println("Метод Симпсона:");
        System.out.printf("h=%.3f: F=%.8f\n", h1, simpH1);
        System.out.printf("h=%.3f: F=%.8f\n", h2, simpH2);
        System.out.printf("Уточнённое значение: %.8f\n", simpRefined);
        System.out.printf("Погрешность: %.8f\n", simpError);
    }

    // Метод прямоугольников (средних)
    public static double rectangleMethod(Function<Double, Double> f, double a, double b, double h) {
        double sum = 0.0;
        double x = a + h / 2; // Средняя точка первого интервала
        while (x < b) {
            sum += f.apply(x);
            x += h;
        }
        return sum * h;
    }

    // Метод трапеций
    public static double trapezoidalMethod(Function<Double, Double> f, double a, double b, double h) {
        double sum = 0.5 * (f.apply(a) + f.apply(b));
        double x = a + h;
        while (x < b) {
            sum += f.apply(x);
            x += h;
        }
        return sum * h;
    }

    // Метод Симпсона
    public static double simpsonMethod(Function<Double, Double> f, double a, double b, double h) {
        if ((b - a) / h % 2 != 0) {
            throw new IllegalArgumentException("Для метода Симпсона (b-a)/h должно быть четным числом");
        }

        double sum = f.apply(a) + f.apply(b);
        double x = a + h;
        boolean even = false;
        while (x < b) {
            sum += (even ? 2 : 4) * f.apply(x);
            x += h;
            even = !even;
        }
        return sum * h / 3;
    }

    // Уточнение значения интеграла методом Рунге-Ромберга
    public static double rungeRombergRefined(double Ih, double Ih2, double h, double h2, int p) {
        return Ih2 + (Ih2 - Ih) / (Math.pow(h / h2, p) - 1);
    }
}