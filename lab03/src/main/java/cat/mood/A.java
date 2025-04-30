package cat.mood;

import java.util.function.Function;

public class A {
    public record Pair(Function<Double, Double> first, Function<String, String> second) {}

    static double omega(int n, int idx, double[] x) {
        double result = 1;
        for (int i = 0; i < n; ++i) {
            if (i != idx) {
                result *= x[idx] - x[i];
            }
        }

        return result;
    }

    public static double[][] difference(double[] x, double[] y) {
        int n = x.length;
        double[][] table = new double[n][n];


        for (int i = 0; i < n; i++) {
            table[i][0] = y[i];
        }

        for (int j = 1; j < n; j++) {
            for (int i = 0; i < n - j; i++) {
                table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i]);
            }
        }

        return table;
    }

    public static Pair lagrange(double[] x, double[] y) {
        int n = x.length;
        double[] w = new double[n];
        for (int i = 0; i < n; ++i) {
            w[i] = omega(n, i, x);
        }

        Function<String, String> fs = str -> {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < n; ++i) {
                if (y[i] / w[i] > 0) {
                    sb.append("+");
                }
                sb.append(y[i] / w[i]);
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        sb.append("(x ");
                        if (x[j] < 0) {
                            sb.append("+ ").append(x[j]);
                        } else {
                            sb.append("- ").append(x[j]);
                        }
                        sb.append(")");
                    }
                }
                sb.append(" ");
            }
            return sb.toString();
        };

        Function<Double, Double> fd = t -> {
            double f = 0;
            for (int i = 0; i < n; ++i) {
                double fi = y[i] / w[i];
                for (int j = 0; j < n; ++j) {
                    if (i != j) {
                        fi *= t - x[j];
                    }
                }
                f += fi;
            }

            return f;
        };
        return new Pair(fd, fs);
    }

    public static Pair newton(double[] x, double[] y) {
        double[][] d = difference(x, y);
        int n = x.length;

        Function<String, String> fs = str -> {
            StringBuilder sb = new StringBuilder();
            sb.append(y[0]);
            for (int i = 1; i < n; ++i) {
                if (d[0][i] > 0) {
                    sb.append(" + ");
                } else {
                    sb.append(" - ");
                }
                sb.append(Math.abs(d[0][i]));
                for (int j = 0; j < i; ++j) {
                    sb.append("(x ");
                    if (x[i] < 0) {
                        sb.append("+ ");
                    } else {
                        sb.append("- ");
                    }
                    sb.append(Math.abs(x[j])).append(")");
                }
            }
            return sb.toString();
        };

        Function<Double, Double> fd = t -> {
            double f = y[0];
            for (int i = 1; i < n; ++i) {
                double fi = d[0][i];
                for (int j = 0; j < i; ++j) {
                    fi *= t - x[j];
                }
                f += fi;
            }

            return f;
        };

        return new Pair(fd, fs);
    }

    static void solve(double[] x, double[] y, double t, Function<Double, Double> f) {
        var L = lagrange(x, y);
        System.out.println("Многочлен Лагранжа:");
        System.out.println(L.second.apply("x"));
        System.out.println("Многочлен Лангранжа в точке x = " + t + ": " + L.first.apply(t));
        System.out.println("Функция в точке x = " + t + ": " + f.apply(t));
        System.out.println("Погрешность: " + Math.abs(f.apply(t) - L.first.apply(t)));

        System.out.println();

        var N = newton(x, y);
        System.out.println("Многочлен Ньютона:");
        System.out.println(N.second.apply("x"));
        System.out.println("Многочлен Ньютона в точке x = " + t + ": " + N.first.apply(t));
        System.out.println("Функция в точке x = " + t + ": " + f.apply(t));
        System.out.println("Погрешность: " + Math.abs(f.apply(t) - N.first.apply(t)));
    }

    public static void main(String[] args) {
        Function<Double, Double> f = x -> Math.asin(x) + x;
        double[] x1 = {-0.4, -0.1, 0.2, 0.5};
        double[] x2 = {-0.4, 0, 0.2, 0.5};
        double[] y1 = new double[x1.length];
        double[] y2 = new double[x2.length];
        for (int i = 0; i < x1.length; ++i) {
            y1[i] = f.apply(x1[i]);
            y2[i] = f.apply(x2[i]);
        }

        double t = 0.1;

        System.out.println("а)");
        solve(x1, y1, t, f);
        System.out.println("б)");
        solve(x2, y2, t, f);
    }
}
