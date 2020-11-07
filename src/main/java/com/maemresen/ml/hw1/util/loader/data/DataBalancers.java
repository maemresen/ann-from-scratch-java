package com.maemresen.ml.hw1.util.loader.data;

import com.maemresen.ml.hw1.util.ann.DataSet;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/**
 * @author Emre Sen
 * @date Dec 30, 2018
 * @contact maemresen07@gmail.com
 */
public class DataBalancers {


    public static DataSet limitToAvg(DataSet dataSet) {

        RealMatrix dataMatrix = dataSet.getDataMatrix();
        double[] values = dataSet.getValues();
        int m = dataSet.getSampleSize();
        int n = dataSet.getFeatureSize();
        double[] classes = dataSet.getClasses();
        double[] numOfClassInstances = dataSet.getNumOfClassInstances();
        int numOfClasses = dataSet.getNumOfClasses();
        double avg = 0.0;
        for (int c = 0; c < numOfClasses; c++) {
            avg += numOfClassInstances[c] / ((double) numOfClasses);
        }

        List<Integer> selectedRowsList = new ArrayList<>();
        numOfClassInstances = new double[numOfClasses];
        for (int i = 0; i < m; i++) {
            double value = values[i];
            for (int c = 0; c < numOfClasses; c++) {
                if (value == classes[c]) {
                    if (numOfClassInstances[c] <= avg) {
                        selectedRowsList.add(i);
                        numOfClassInstances[c]++;
                        break;
                    }
                }
            }
        }

        int[] selectedRows = new int[selectedRowsList.size()];
        int[] selectedColumns = IntStream.range(0, n + 1).toArray();
        for (int r = 0; r < selectedRowsList.size(); r++) {
            selectedRows[r] = selectedRowsList.get(r);
        }

//        System.out.println(selectedRows.length);
//        System.out.println(selectedColumns.length);


        RealMatrix balancedDataMatrix = dataMatrix.getSubMatrix(selectedRows
                , selectedColumns);
        return new DataSet(balancedDataMatrix.getData());

    }

    public static DataSet duplicateAboveAvg(DataSet dataSet) {
        double[][] data = dataSet.getData();
        double[] values = dataSet.getValues();
        int n = dataSet.getFeatureSize();
        int m = dataSet.getSampleSize();

        double[] classes = dataSet.getClasses();
        double[] numOfClassInstances = dataSet.getNumOfClassInstances();
        int numOfClasses = dataSet.getNumOfClasses();
        double avg = 0.0;
        for (int c = 0; c < numOfClasses; c++) {
            avg += numOfClassInstances[c] / ((double) numOfClasses);
        }


        List<double[]> dataList = new ArrayList<>();
        for (int i = 0; i < dataSet.getSampleSize(); i++) {
            dataList.add(data[i]);
            double value = values[i];
            for (int c = 0; c < numOfClasses; c++) {
                if (value == classes[c]) {
                    if (numOfClassInstances[c] <= avg) {
                        dataList.add(data[i]);
                    }
                }
            }
        }

        double[][] result = new double[dataList.size()][n + 1];
        for (int i = 0; i < result.length; i++) {
            result[i] = dataList.get(i);
        }
        return new DataSet(result);
    }

    public static DataSet duplicateMin(DataSet dataSet) {
        double[][] data = dataSet.getData();
        double[] values = dataSet.getValues();
        int n = dataSet.getFeatureSize();
        int m = dataSet.getSampleSize();

        double[] classes = dataSet.getClasses();
        double[] numOfClassInstances = dataSet.getNumOfClassInstances();
        int numOfClasses = dataSet.getNumOfClasses();


        int mini = 0;
        for (int c = 0; c < numOfClasses; c++) {
            if (numOfClassInstances[c] <= numOfClassInstances[mini]) {
                mini = c;
            }
        }

        List<double[]> dataList = new ArrayList<>();
        for (int i = 0; i < dataSet.getSampleSize(); i++) {
            dataList.add(data[i]);
            double value = values[i];
            if (value == classes[mini]) {
                dataList.add(data[i]);
            }
        }

        double[][] result = new double[dataList.size()][n + 1];
        for (int i = 0; i < result.length; i++) {
            result[i] = dataList.get(i);
        }
        return new DataSet(result);
    }
}
