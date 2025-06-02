package cat.mood;

public class Complex {
    double real;
    double imaginary;

    public Complex(double real, double imaginary) {
        this.real = real;
        this.imaginary = imaginary;
    }

    @Override
    public String toString() {
        if (imaginary == 0) {
            return real + "";
        }
        return real + (imaginary > 0 ? " + " : " - ") + Math.abs(imaginary) + "i";
    }

    public static Complex subtraction(Complex a, Complex b) {
        return new Complex(a.real - b.real, a.imaginary - b.imaginary);
    }
}
