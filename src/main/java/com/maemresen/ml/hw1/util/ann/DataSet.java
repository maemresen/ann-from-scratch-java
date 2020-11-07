package com.maemresen.ml.hw1.util.ann;

import com.maemresen.ml.hw1.util.helper.ArrayHelper;
import com.maemresen.ml.hw1.util.helper.MatrixHelper;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Model that Represents a DataSet for a Problem.
 * <p>
 * Features and Values are extracting.
 * Sample and Feature sizes calculated.
 * Means and Ranges of Features calculated
 * Number of Distinct classes claculated
 *
 * @author Emre Sen
 * @date Dec 26, 2018
 * @contact maemresen07@gmail.com
 */
public class DataSet implements Serializable {

    private static final long serialVersionUID = -299482035708790407L;
    private static final Logger LOGGER = LoggerFactory.getLogger(DataSet.class);

    private final RealMatrix dataMatrix;
    private final RealMatrix featuresMatrix;
    private final RealMatrix valuesMatrix;

    private final double[] values;
    private final double[] numOfClassInstances;
    private final double[] classes;

    private final int n;
    private final int m;
    private final int numOfClasses;

    private RealMatrix means;
    private RealMatrix ranges;

    public DataSet(double[][] d) {

        LOGGER.trace("Start Creating Data Matrix");
        this.dataMatrix = MatrixUtils.createRealMatrix(d);
        LOGGER.trace("Finish Creating Data Matrix");

        LOGGER.trace("Start Finding Sample and Feature sizes");
        n = this.dataMatrix.getColumnDimension() - 1;
        m = this.dataMatrix.getRowDimension();
        LOGGER.trace("Finish Finding Sample and Feature sizes");

        // featuresMatrix
        LOGGER.trace("Start Creating Feature Matrix");
        featuresMatrix = dataMatrix.getSubMatrix(0, m - 1, 0, n - 1);
        LOGGER.trace("Finish Creating Feature Matrix");

        LOGGER.trace("Start Calculating Means and Ranges of Features");
        means = MatrixHelper.meanRowMatrix(featuresMatrix);
        ranges = MatrixHelper.rangeRowMatrix(featuresMatrix);
        LOGGER.trace("Finish Calculating Means and Ranges of Features");

        // valuesMatrix
        LOGGER.trace("Start Loading Classes from DataSet");
        values = dataMatrix.getColumn(n);
        LOGGER.trace("Finish Loading Classes from DataSet");

        LOGGER.trace("Start Finding Class Types");
        classes = ArrayHelper.getUniqueValues(values);    // class types
        numOfClasses = classes.length;
        LOGGER.trace("Finish Finding Class Types");

        LOGGER.trace("Start Finding Number of Class Instances");
        numOfClassInstances = new double[numOfClasses];
        for (double value : values) {
            int c = 0;
            while (value != classes[c]) {
                c++;
            }
            numOfClassInstances[c]++;
        }
        LOGGER.trace("Finish Finding Number of Class Instances");


        LOGGER.trace("Start Creating Values Matrix");
        valuesMatrix = MatrixUtils.createRealMatrix(m, numOfClasses);
        for (
                int i = 0;
                i < m; i++) {
            double value = values[i];
            for (int j = 0; j < numOfClasses; j++) {
                if (classes[j] == value) {
                    valuesMatrix.setEntry(i, j, 1);
                    break;
                }
            }
        }
        LOGGER.trace("Finish Creating Values Matrix");
    }

    /* Getters/Setters */

    // ..
    public RealMatrix getDataMatrix() {
        return dataMatrix;
    }

    public RealMatrix getFeaturesMatrix() {
        return featuresMatrix;
    }

    public RealMatrix getValuesMatrix() {
        return valuesMatrix;
    }

    // ..
    public double[] getValues() {
        return values;
    }

    public double[] getNumOfClassInstances() {
        return numOfClassInstances;
    }

    public double[] getClasses() {
        return classes;
    }

    // ..
    public int getFeatureSize() {
        return n;
    }

    public int getSampleSize() {
        return m;
    }

    public int getNumOfClasses() {
        return numOfClasses;
    }

    /**/
    public double[][] getData() {
        return dataMatrix.getData();
    }
    /**/

    /**
     * Normalize input according to the dataset.
     * Subtract means from each feature than divide by range (max - min)
     *
     * @param input each row corresponds sample
     *              each column corresponds featuresMatrix of one sample
     * @return normalized feature valuesMatrix
     */
    public RealMatrix applyMeanNormalization(RealMatrix input) {
        LOGGER.trace("Start Applying Mean Normaliztion");
        RealMatrix m = MatrixHelper.scaleRowMatrix(this.means, input.getRowDimension());
        RealMatrix r = MatrixHelper.scaleRowMatrix(this.ranges, input.getRowDimension());
        RealMatrix result = MatrixHelper.dotDivision(input.subtract(m), r);
        LOGGER.trace("Finish Applying Mean Normaliztion");
        return result;
    }

    public void print() {
        LOGGER.info("Classes in DataSet : " + Arrays.toString(classes));
        LOGGER.info("Number of Samples of the Classes in DataSet : " + Arrays.toString(numOfClassInstances));
    }
}
