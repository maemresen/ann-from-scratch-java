package com.maemresen.ml.hw1.util.ann;

import com.maemresen.ml.hw1.util.helper.ArrayHelper;
import com.maemresen.ml.hw1.util.helper.MatrixHelper;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;

/**
 * Model that Represents an Artificial Neural Network Layer
 *
 * @author Emre Sen
 * @date Dec 26, 2018
 * @contact maemresen07@gmail.com
 */
public class Layer implements Serializable {

    public enum LayerType {

        INPUT, HIDDEN, OUTPUT
    }


    private static final long serialVersionUID = -299482035708790407L;
    private static final Logger LOGGER = LoggerFactory.getLogger(Layer.class);

    private String layerName;
    private Layer nextLayer;

    private RealMatrix thetas;
    private RealMatrix activations;

    private int numOfUnits;

    private LayerType layerType;

    Layer(String layerName, int numOfUnits, LayerType layerType) {
        this.layerName = layerName;
        this.numOfUnits = numOfUnits;
        this.layerType = layerType;
    }

    /**
     * To Initialize theta values (weights) for current ANN Layer
     */
    public void initThetas() {
        if (isOutputLayer()) {
            LOGGER.trace("Output Layer does not have theta values");
            return;
        }
        LOGGER.trace("Start Initializing theta values for " + layerName);
        thetas = MatrixUtils.createRealMatrix(ArrayHelper.generateRandomMatrix(nextLayer.numOfUnits, this.numOfUnits));
        LOGGER.trace("Finish Initializing theta values for " + layerName);
    }


    /*
    -------------------------------
        Getters / Setters
    -------------------------------
    */


    public String getLayerName() {
        return layerName;
    }

    public void setLayerName(String layerName) {
        this.layerName = layerName;
    }

    public Layer getNextLayer() {
        return nextLayer;
    }

    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
        this.thetas = MatrixHelper.randomMatrix(nextLayer.numOfUnits, this.numOfUnits);
    }

    public RealMatrix getThetas() {
        return thetas;
    }

    public void setThetas(RealMatrix thetas) {
        this.thetas = thetas;
    }

    public RealMatrix getActivations() {
        return activations;
    }

    public void setActivations(RealMatrix activations) {
        this.activations = activations;
    }

    public int getNumOfUnits() {
        return numOfUnits;
    }

    public void setNumOfUnits(int numOfUnits) {
        this.numOfUnits = numOfUnits;
    }

    public LayerType getLayerType() {
        return layerType;
    }

    public void setLayerType(LayerType layerType) {
        this.layerType = layerType;
    }


    /*
    -------------------------------
        Forward Propagation
    -------------------------------
    */

    /**
     * Forwarding activations of current layer to the next layer
     *
     * @param input activations of current layer calculated by previous layer activations and theta values (weights)
     * @return output of the ANN for given input value
     */
    public RealMatrix forward(RealMatrix input) {
        this.activations = input;
        if (isOutputLayer()) {
            LOGGER.trace("Hypothesis Found");
            return input;
        }
        LOGGER.trace("Forwarding Data " + layerName + " -> " + nextLayer.layerName);
        RealMatrix layerZ = input.multiply(thetas.transpose());
        return nextLayer.forward(MatrixHelper.sigmoid(layerZ));
    }


    /*
    -------------------------------
        Back Propagation
    -------------------------------
    */

    /**
     * Applying BackPropagation for calculating gradient for each layer
     * <p>
     * Updating theta values (weights).
     *
     * @param values real values of samples
     * @param m      sample size
     * @param lambda regularization term for bp
     * @param alpha  learning rate for bp
     * @return current layers error (delta term)
     */
    public RealMatrix updateThetas(RealMatrix values, int m, double alpha, Double lambda) {
        if (nextLayer == null) {
            return activations.subtract(values);
        }
        LOGGER.trace("Start Updating " + layerName + " Thetas (Weights)");
        LOGGER.trace("Start Calculating Next Layer : " + nextLayer.getLayerName() + " Delta");
        RealMatrix nextLayerDelta = nextLayer.updateThetas(values, m, alpha, lambda);
        LOGGER.trace("Finish Calculating Next Layer : " + nextLayer.getLayerName() + " Delta");

        LOGGER.trace("Start Calculating " + nextLayer.getLayerName() + " Derivative");
        RealMatrix derivative = MatrixHelper.dotProduct(activations, activations.scalarMultiply(-1).scalarAdd(1));
        RealMatrix leftTerm = nextLayerDelta.multiply(thetas);
        LOGGER.trace("Finish Calculating " + nextLayer.getLayerName() + " Derivative");

        LOGGER.trace("Start Calculating " + layerName + " Delta");
        RealMatrix layerDelta = MatrixHelper.dotProduct(leftTerm, derivative);
        LOGGER.trace("Finish Calculating " + layerName + " Delta");

        LOGGER.trace("Start Calculating " + layerName + " Capital Delta");
        RealMatrix layerCapitalDelta = activations.transpose().multiply(nextLayerDelta).transpose();
        LOGGER.trace("Finish Calculating " + layerName + " Capital Delta");

        int thetasRow = thetas.getRowDimension();
        int thetasCol = thetas.getColumnDimension();
        if (lambda != null) {
            LOGGER.trace("Start Adding Regularization Term to Gradient");
            RealMatrix otherThetas = thetas.getSubMatrix(0, thetasRow - 1, 1, thetasCol - 1);
            RealMatrix tmp = MatrixUtils.createRealMatrix(thetasRow, thetasCol);
            tmp.setSubMatrix(otherThetas.scalarMultiply(lambda).getData(), 0, 1);
            layerCapitalDelta.add(tmp);
            LOGGER.trace("Finish Adding Regularization Term to Gradient");
        }


        LOGGER.trace("Start Calculating " + layerName + " Gradient");
        double scalar = (alpha / (2.0 * m));
        RealMatrix layerGradient = layerCapitalDelta.scalarMultiply(scalar);
        LOGGER.trace("Finish Calculating " + layerName + " Gradient");
        thetas = thetas.subtract(layerGradient);
        LOGGER.trace("Finish Updating " + layerName + " Thetas (Weights)");
        return layerDelta;
    }

    public double sumThetaSquares() {
        if (isOutputLayer()) {
            return 0;
        }
        LOGGER.trace("Start Summing " + layerName + " Thetas for Regularization");
        double result = Math.sqrt(MatrixHelper.sum(thetas)) + nextLayer.sumThetaSquares();
        LOGGER.trace("Finish Summing " + layerName + " Thetas for Regularization");
        return result;
    }

    /**/
    public boolean isOutputLayer() {
        return layerType == LayerType.OUTPUT;
    }

    /**/

    public void print() {
        LOGGER.info(layerName);
        LOGGER.info(String.format(" Name   : %s", layerName));
        LOGGER.info(String.format(" Units  : %s", numOfUnits));
        LOGGER.info(String.format(" Type   : %s", layerType));
        if (thetas != null) {
            LOGGER.info(String.format(" Thetas : [%s x %s]"
                    , thetas.getRowDimension()
                    , thetas.getColumnDimension()));
        }
    }
}