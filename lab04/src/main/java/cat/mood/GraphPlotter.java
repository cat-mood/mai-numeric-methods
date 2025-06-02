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

public class GraphPlotter {

    public static void plotAllSolutions(double[][] eulerSolution, double[][] rkSolution,
                                        double[][] adamsSolution) {
        // Создаем набор данных
        XYSeriesCollection dataset = new XYSeriesCollection();

        // Добавляем точное решение (берем из любого решения, они одинаковые)
        XYSeries exactSolution = new XYSeries("Точное решение");
        for (double[] row : eulerSolution) {
            exactSolution.add(row[0], row[3]);
        }
        dataset.addSeries(exactSolution);

        // Добавляем решение методом Эйлера
        XYSeries eulerSeries = new XYSeries("Метод Эйлера");
        for (double[] row : eulerSolution) {
            eulerSeries.add(row[0], row[1]);
        }
        dataset.addSeries(eulerSeries);

        // Добавляем решение методом Рунге-Кутты
        XYSeries rkSeries = new XYSeries("Метод Рунге-Кутты");
        for (double[] row : rkSolution) {
            rkSeries.add(row[0], row[1]);
        }
        dataset.addSeries(rkSeries);

        // Добавляем решение методом Адамса
        XYSeries adamsSeries = new XYSeries("Метод Адамса");
        for (double[] row : adamsSolution) {
            adamsSeries.add(row[0], row[1]);
        }
        dataset.addSeries(adamsSeries);

        // Создаем график
        JFreeChart chart = ChartFactory.createXYLineChart(
                "Сравнение численных методов", // Заголовок
                "x", // Ось X
                "y", // Ось Y
                dataset, // Данные
                PlotOrientation.VERTICAL,
                true, // Показывать легенду
                true, // Использовать подсказки
                false // URLs
        );

        // Настраиваем внешний вид
        XYPlot plot = (XYPlot) chart.getPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();

        // Настраиваем цвета и стили линий
        renderer.setSeriesPaint(0, Color.BLACK);    // Точное решение
        renderer.setSeriesPaint(1, Color.RED);      // Эйлер
        renderer.setSeriesPaint(2, Color.BLUE);     // Рунге-Кутта
        renderer.setSeriesPaint(3, Color.GREEN);    // Адамс

        // Настраиваем толщину линий
        for (int i = 0; i < dataset.getSeriesCount(); i++) {
            renderer.setSeriesStroke(i, new BasicStroke(2.0f));
            renderer.setSeriesShapesVisible(i, false);
        }

        plot.setRenderer(renderer);

        // Создаем окно для отображения графика
        JFrame frame = new JFrame("Сравнение численных методов");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new ChartPanel(chart));
        frame.setSize(800, 600);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}
