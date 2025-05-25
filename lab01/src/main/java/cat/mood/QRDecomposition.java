package cat.mood;

import java.util.ArrayList;
import java.util.List;

public class QRDecomposition {

    public static void main(String[] args) {
        List<List<Complex>> A = new ArrayList<>();
        A.add(List.of(new Complex(0), new Complex(-1), new Complex(3)));
        A.add(List.of(new Complex(-1), new Complex(6), new Complex(-3)));
        A.add(List.of(new Complex(-8), new Complex(4), new Complex(2)));

        QRResult qrResult = QR(A);
        List<List<Complex>> Q = qrResult.Q;
        List<List<Complex>> R = qrResult.R;

        System.out.println("Матрица Q:");
        printMatrix(Q);

        System.out.println("\nМатрица R:");
        printMatrix(R);

        System.out.println("\nПроверка QR (Q * R):");
        printMatrix(matrixMultiply(Q, R));

        EigenResult eigenResult = eigenvalues(A, 1e-15, 1000);
        List<Complex> eigenvals = eigenResult.eigenvalues;
        int iterations = eigenResult.iterations;

        System.out.println("\nСобственные значения:");
        for (Complex val : eigenvals) {
            System.out.println(val);
        }
    }

    static class Complex {
        double re;
        double im;

        Complex(double re) {
            this(re, 0);
        }

        Complex(double re, double im) {
            this.re = re;
            this.im = im;
        }

        Complex add(Complex other) {
            return new Complex(this.re + other.re, this.im + other.im);
        }

        Complex subtract(Complex other) {
            return new Complex(this.re - other.re, this.im - other.im);
        }

        Complex multiply(Complex other) {
            return new Complex(
                    this.re * other.re - this.im * other.im,
                    this.re * other.im + this.im * other.re
            );
        }

        Complex divide(double scalar) {
            return new Complex(this.re / scalar, this.im / scalar);
        }

        Complex conjugate() {
            return new Complex(this.re, -this.im);
        }

        double abs() {
            return Math.sqrt(re * re + im * im);
        }

        @Override
        public String toString() {
            if (im == 0) return String.format("%.4f", re);
            return String.format("%.4f%+.4fi", re, im);
        }
    }

    static class QRResult {
        List<List<Complex>> Q;
        List<List<Complex>> R;

        QRResult(List<List<Complex>> Q, List<List<Complex>> R) {
            this.Q = Q;
            this.R = R;
        }
    }

    static class EigenResult {
        List<Complex> eigenvalues;
        int iterations;

        EigenResult(List<Complex> eigenvalues, int iterations) {
            this.eigenvalues = eigenvalues;
            this.iterations = iterations;
        }
    }

    public static void printMatrix(List<List<Complex>> A) {
        int n = A.size();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.print(A.get(i).get(j) + "\t");
            }
            System.out.println();
        }
    }

    public static double vecNorm(List<Complex> v) {
        double sum = 0;
        for (Complex x : v) {
            sum += x.abs() * x.abs();
        }
        return Math.sqrt(sum);
    }

    public static Complex dotProduct(List<Complex> a, List<Complex> b) {
        if (a.size() != b.size()) {
            throw new IllegalArgumentException("Векторы должны быть одного размера");
        }
        Complex sum = new Complex(0);
        for (int i = 0; i < a.size(); i++) {
            sum = sum.add(a.get(i).multiply(b.get(i).conjugate()));
        }
        return sum;
    }

    public static List<List<Complex>> matrixMultiply(List<List<Complex>> A, List<List<Complex>> B) {
        int rowsA = A.size();
        int colsA = A.get(0).size();
        int rowsB = B.size();
        int colsB = B.get(0).size();

        if (colsA != rowsB) {
            throw new IllegalArgumentException("Несоответствие размеров матриц");
        }

        List<List<Complex>> result = new ArrayList<>();
        for (int i = 0; i < rowsA; i++) {
            List<Complex> row = new ArrayList<>();
            for (int j = 0; j < colsB; j++) {
                Complex sum = new Complex(0);
                for (int k = 0; k < colsA; k++) {
                    sum = sum.add(A.get(i).get(k).multiply(B.get(k).get(j)));
                }
                row.add(sum);
            }
            result.add(row);
        }
        return result;
    }

    public static double maxSubdiagonal(List<List<Complex>> A) {
        int n = A.size();
        double maxVal = 0;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                maxVal = Math.max(maxVal, A.get(i).get(j).abs());
            }
        }
        return maxVal;
    }

    public static QRResult QR(List<List<Complex>> matr) {
        int n = matr.size();
        List<List<Complex>> A = new ArrayList<>();
        for (List<Complex> row : matr) {
            List<Complex> newRow = new ArrayList<>();
            for (Complex val : row) {
                newRow.add(new Complex(val.re, val.im));
            }
            A.add(newRow);
        }

        List<List<Complex>> Q = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<Complex> row = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                row.add(i == j ? new Complex(1) : new Complex(0));
            }
            Q.add(row);
        }

        for (int j = 0; j < n; j++) {
            List<Complex> a = new ArrayList<>();
            for (int i = j; i < n; i++) {
                a.add(A.get(i).get(j));
            }

            List<Complex> v = new ArrayList<>();
            for (int i = 0; i < a.size(); i++) {
                v.add(new Complex(0));
            }

            if (a.get(0).abs() != 0) {
                double sign = a.get(0).re >= 0 ? 1 : -1;
                v.set(0, a.get(0).add(new Complex(sign * vecNorm(a), 0)));
            } else {
                v.set(0, new Complex(vecNorm(a), 0));
            }

            for (int i = 1; i < a.size(); i++) {
                v.set(i, a.get(i));
            }

            double normV = vecNorm(v);
            if (normV == 0) {
                break;
            }

            for (int i = 0; i < v.size(); i++) {
                v.set(i, v.get(i).divide(normV));
            }

            List<List<Complex>> H = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                List<Complex> row = new ArrayList<>();
                for (int k = 0; k < n; k++) {
                    row.add(i == k ? new Complex(1) : new Complex(0));
                }
                H.add(row);
            }

            for (int i = j; i < n; i++) {
                for (int k = j; k < n; k++) {
                    Complex term = v.get(i - j).multiply(v.get(k - j).conjugate());
                    H.get(i).set(k, H.get(i).get(k).subtract(term.multiply(new Complex(2))));
                }
            }

            Q = matrixMultiply(Q, H);
            A = matrixMultiply(H, A);
        }

        return new QRResult(Q, A);
    }

    public static EigenResult eigenvalues(List<List<Complex>> matr, double eps, int maxIters) {
        int n = matr.size();
        List<List<Complex>> A = new ArrayList<>();
        for (List<Complex> row : matr) {
            List<Complex> newRow = new ArrayList<>();
            for (Complex val : row) {
                newRow.add(new Complex(val.re, val.im));
            }
            A.add(newRow);
        }

        int iterations = 0;
        for (iterations = 0; iterations < maxIters; iterations++) {
            Complex mu = A.get(n - 1).get(n - 1);

            // Сдвиг (shift)
            for (int i = 0; i < n; i++) {
                A.get(i).set(i, A.get(i).get(i).subtract(mu));
            }

            QRResult qrResult = QR(A);
            List<List<Complex>> Q = qrResult.Q;
            List<List<Complex>> R = qrResult.R;

            A = matrixMultiply(R, Q);

            // Обратный сдвиг
            for (int i = 0; i < n; i++) {
                A.get(i).set(i, A.get(i).get(i).add(mu));
            }

            double maxSubdiag = maxSubdiagonal(A);
            if (maxSubdiag < eps) {
                break;
            }
        }

        List<Complex> eigenvals = new ArrayList<>();
        int i = 0;
        while (i < n) {
            if (i == n - 1 || A.get(i + 1).get(i).abs() < eps) {
                eigenvals.add(A.get(i).get(i));
                i++;
            } else {
                Complex a = A.get(i).get(i);
                Complex b = A.get(i).get(i + 1);
                Complex c = A.get(i + 1).get(i);
                Complex d = A.get(i + 1).get(i + 1);

                Complex trace = a.add(d);
                Complex det = a.multiply(d).subtract(b.multiply(c));

                Complex discriminant = trace.multiply(trace).subtract(det.multiply(new Complex(4)));
                Complex sqrtDiscriminant = sqrt(discriminant);

                Complex r1 = trace.add(sqrtDiscriminant).divide(2);
                Complex r2 = trace.subtract(sqrtDiscriminant).divide(2);

                eigenvals.add(r1);
                eigenvals.add(r2);
                i += 2;
            }
        }

        return new EigenResult(eigenvals, iterations + 1);
    }

    private static Complex sqrt(Complex c) {
        double r = Math.sqrt(c.re * c.re + c.im * c.im);
        double theta = Math.atan2(c.im, c.re) / 2;
        return new Complex(r * Math.cos(theta), r * Math.sin(theta));
    }
}