package com.maemresen.ml.hw1.util.helper;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

/**
 * @author Emre Sen
 * @date Dec 26, 2018
 * @contact maemresen07@gmail.com
 */
public class MatrixHelper {

    /* Utils */
    public static RealMatrix dotProduct(RealMatrix matrix1, RealMatrix matrix2) {
        return MatrixUtils.createRealMatrix(ArrayHelper.multiply(matrix1.getData(), matrix2.getData()));
    }

    public static RealMatrix dotDivision(RealMatrix matrix1, RealMatrix matrix2) {
        return MatrixUtils.createRealMatrix(ArrayHelper.divide(matrix1.getData(), matrix2.getData()));
    }

    public static RealMatrix sigmoid(RealMatrix matrix) {
        return MatrixUtils.createRealMatrix(ArrayHelper.sigmoid(matrix.getData()));
    }

    public static RealMatrix log(RealMatrix matrix) {
        return MatrixUtils.createRealMatrix(ArrayHelper.log(matrix.getData()));
    }

    public static double sum(RealMatrix matrix) {
        return ArrayHelper.sum(matrix.getData());
    }

    /* Helpers */
    public static RealMatrix meanRowMatrix(RealMatrix matrix) {
        return MatrixUtils.createRowRealMatrix(ArrayHelper.mean(matrix.getData()));
    }

    public static RealMatrix rangeRowMatrix(RealMatrix matrix) {
        return MatrixUtils.createRowRealMatrix(ArrayHelper.range(matrix.getData()));
    }

    /**/
    public static RealMatrix scaleRowMatrix(RealMatrix matrix, int rowLen) {
        return MatrixUtils.createRealMatrix(ArrayHelper.scaleArray(matrix.getData()[0], rowLen));
    }

    /**/
    public static RealMatrix addBiasTermToMatrix(RealMatrix matrix) {
        int m = matrix.getRowDimension();
        int n = matrix.getColumnDimension();
        RealMatrix result = MatrixUtils.createRealMatrix(m, n + 1).scalarAdd(1);
        result.setSubMatrix(matrix.getData(), 0, 1);
        return result;
    }

    /**/
    public static RealMatrix max(RealMatrix matrix) {
        return MatrixUtils.createRealMatrix(ArrayHelper.max(matrix.getData()));
    }

    /**/
    public static RealMatrix randomMatrix(int rowLen, int colLen){
        return MatrixUtils.createRealMatrix(ArrayHelper.generateRandomMatrix(rowLen, colLen));
    }
}
