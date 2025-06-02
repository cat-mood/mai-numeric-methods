package cat.mood;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import javax.swing.JFrame;
import java.awt.Color;
import java.awt.BasicStroke;

public class BVPGraphPlotter {

    public static void plotSolutions(double[][] solution, String methodName) {
        // Создаем набор данных
        XYSeriesCollection dataset = new XYSeriesCollection();

        // Добавляем численное решение
        XYSeries numericalSolution = new XYSeries("Численное решение");
        for (double[] row : solution) {
            numericalSolution.add(row[0], row[1]); // x и y численного метода
        }
        dataset.addSeries(numericalSolution);

        // Добавляем точное решение
        XYSeries exactSolution = new XYSeries("Точное решение");
        for (double[] row : solution) {
            exactSolution.add(row[0], row[3]); // x и точное решение
        }
        dataset.addSeries(exactSolution);

        // Создаем график
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Решение краевой задачи - " + methodName,
                "x",
                "y",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        // Настраиваем внешний вид
        XYPlot plot = (XYPlot) chart.getPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();

        // Настраиваем цвета и стили линий
        renderer.setSeriesPaint(0, Color.RED);     // Численное решение
        renderer.setSeriesPaint(1, Color.BLUE);    // Точное решение

        renderer.setSeriesStroke(0, new BasicStroke(2.0f));
        renderer.setSeriesStroke(1, new BasicStroke(2.0f));

        // Отключаем отображение точек
        renderer.setSeriesShapesVisible(0, false);
        renderer.setSeriesShapesVisible(1, false);

        plot.setRenderer(renderer);

        // Создаем окно для отображения графика
        JFrame frame = new JFrame("Краевая задача - " + methodName);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.setSize(800, 600);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    public static void plotError(double[][] solution, String methodName) {
        XYSeriesCollection dataset = new XYSeriesCollection();

        // Добавляем график погрешности
        XYSeries errorSeries = new XYSeries("Погрешность");
        for (double[] row : solution) {
            errorSeries.add(row[0], row[4]); // x и погрешность
        }
        dataset.addSeries(errorSeries);

        // Создаем график
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Погрешность метода - " + methodName,
                "x",
                "Погрешность",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        // Настраиваем внешний вид
        XYPlot plot = (XYPlot) chart.getPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();

        renderer.setSeriesPaint(0, Color.RED);
        renderer.setSeriesStroke(0, new BasicStroke(2.0f));
        renderer.setSeriesShapesVisible(0, false);

        plot.setRenderer(renderer);

        // Создаем окно для отображения графика погрешности
        JFrame frame = new JFrame("Погрешность - " + methodName);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.setSize(800, 600);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
