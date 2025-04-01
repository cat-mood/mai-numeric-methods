package cat.mood;

import java.util.function.Function;

public class A {
    public static double derivative(Function<Double, Double> f, double x, double eps) {
        double dy = f.apply(x + eps) - f.apply(x);
        return dy / eps;
    }

    public static double secondDerivative(Function<Double, Double> f, double x, double eps) {
        double dy = f.apply(x + eps) - 2 * f.apply(x) + f.apply(x - eps);
        return dy / (eps * eps);
    }

    public static boolean checkFunction(Function<Double, Double> phi, double eps, double a, double b) {
        double x = a;
        while (x < b) {
            double y = phi.apply(x);
            if (y < a || y > b) {
                return false;
            }
            if (Math.abs(derivative(phi, x, eps)) >= 1) {
                return false;
            }
            x += eps;
        }

        return true;
    }

    public static double iteration(Function<Double, Double> phi, double eps, double a, double b) {
        boolean check = checkFunction(phi, eps, a, b);
        if (!check) {
            throw new RuntimeException("Не выполнено условие сходимости");
        }

        double prev = a;
        double cur = phi.apply(prev);
        while (Math.abs(cur - prev) > eps) {
            prev = cur;
            cur = phi.apply(prev);
        }

        return cur;
    }

    public static double newton(Function<Double, Double> f, double eps, double a, double b) {
        if (f.apply(a) * f.apply(b) >= 0) {
            throw new RuntimeException("Не выполнено условие сходимости");
        }

        double prev = a;
        while (prev < b) {
            if (f.apply(prev) * secondDerivative(f, prev, eps) > 0) {
                break;
            }
            prev += eps;
        }
        if (f.apply(prev) * secondDerivative(f, prev, eps) <= 0) {
            throw new RuntimeException("Не выполнено условие сходимости");
        }
        double cur = prev - f.apply(prev) / derivative(f, prev, eps);
        while (Math.abs(cur - prev) > eps) {
            prev = cur;
            cur = prev - f.apply(prev) / derivative(f, prev, eps);
        }

        return cur;
    }

    public static void main(String[] args) {
        System.out.println("Метод простой итерации:");
        System.out.println(iteration(x -> (Math.pow(2 * x + 1, 0.25)), 0.0001, 0, 2));
        System.out.println("Метод Ньютона:");
        System.out.println(newton(x -> (Math.pow(x, 4) - 2 * x - 1), 0.0001, 0, 2));
    }
}
