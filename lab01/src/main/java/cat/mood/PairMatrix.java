package cat.mood;

public class PairMatrix {
    public double[][] first;
    public double[][] second;

    public PairMatrix(double[][] first, double[][] second) {
        this.first = MatrixUtils.copy2DArray(first);
        this.second = MatrixUtils.copy2DArray(second);
    }
}
