package com.maemresen.ml.hw1.util.helper;

import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

/**
 * @author Emre Sen
 * @date Dec 22, 2018
 * @contact maemresen07@gmail.com
 */
public class ArrayHelper {

    public static double[][] listTo2DDoubleArray(List<List<String>> list) {
        double[][] result = new double[list.size()][list.get(0).size()];
        int i = 0;
        for (List<String> sample : list) {
            int j = 0;
            for (String feature : sample) {
                result[i][j++] = Double.parseDouble(feature);
                if (j == 0) {
                    System.out.println(feature);
                }
            }
            i++;
        }
        return result;
    }

    public static double[] getUniqueValues(double[] data) {
        return Arrays.stream(data).distinct().sorted().toArray();
    }

    public static double[][] generateRandomMatrix(int r, int c) {
        Random rand = new Random();
        double[][] matrix = new double[r][c];
        for (double[] row : matrix) {
            int l = row.length;
            for (int i = 0; i < l; i++) {
                row[i] = rand.nextDouble();
            }
        }
        return matrix;

    }

    public static double[][] sigmoid(double[][] values) {
        return applyFunction(values, MathHelper::sigmoid);
    }

    public static double[][] log(double[][] values) {
        return applyFunction(values, MathHelper::log);
    }

    private static double[][] applyFunction(double[][] values, Function<Double, Double> function) {
        for (double[] row : values) {
            int len = row.length;
            for (int j = 0; j < len; j++) {
                row[j] = function.apply(row[j]);
            }
        }
        return values;
    }

    /**/
    public static double sum(double[][] values) {
        return Arrays.stream(values).flatMapToDouble(Arrays::stream).sum();
    }

    public static double[][] multiply(double[][] matrix1, double[][] matrix2) {
        int m = matrix1.length;
        int n = matrix1[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double a = matrix1[i][j];
                double b = matrix2[i][j];
                matrix1[i][j] = a * b;
            }
        }
        return matrix1;
    }

    public static double[][] divide(double[][] matrix1, double[][] matrix2) {
        int m = matrix1.length;
        int n = matrix1[0].length;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double a = matrix1[i][j];
                double b = matrix2[i][j];
                if (b == 0) {
                    continue;
                }
                matrix1[i][j] = a / b;
            }
        }
        return matrix1;
    }

    /**/
    public static double[] mean(double[][] matrix) {
        int rowLen = matrix.length;
        int colLen = matrix[0].length;
        double[] mean = new double[colLen];
        for (double[] matrix1 : matrix) {
            for (int col = 0; col < colLen; col++) {
                double val = matrix1[col];
                mean[col] += (val / (double) rowLen);
            }
        }
        return mean;
    }

    public static double[] range(double[][] matrix) {
        int rowLen = matrix.length;
        int colLen = matrix[0].length;
        double[] colMax = new double[colLen];
        double[] colMin = new double[colLen];
        for (double[] matrix1 : matrix) {
            for (int col = 0; col < colLen; col++) {
                double val = matrix1[col];
                if (val > colMax[col]) {
                    colMax[col] = val;
                } else if (val < colMin[col]) {
                    colMin[col] = val;
                }
            }
        }

        double[] range = new double[colLen];
        for (int col = 0; col < colLen; col++) {
            range[col] = colMax[col] - colMin[col];
        }
        return range;
    }

    /**/
    public static double[][] scaleArray(double[] matrix, int rowLen) {
        int colLen = matrix.length;
        double[][] result = new double[rowLen][colLen];
        for (int row = 0; row < rowLen; row++) {
            for (int col = 0; col < colLen; col++) {
                result[row][col] = matrix[col];
            }

        }
        return result;
    }

    /**/
    public static double[][] max(double[][] matrix) {
        int rowLen = matrix.length;
        int colLen = matrix[0].length;
        double[][] result = new double[rowLen][2];
        for (int row = 0; row < rowLen; row++) {
            double max = matrix[row][0];
            double maxi = 0;
            for (int col = 0; col < colLen; col++) {
                double val = matrix[row][col];
                if (val > max) {
                    max = val;
                    maxi = col;
                }
            }
            result[row][0] = maxi + 1;
            result[row][1] = max;
        }
        return result;
    }
}
