package com.maemresen.ml.hw1.util.ann;

import com.maemresen.ml.hw1.util.helper.MatrixHelper;
import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * Model that Represents an Artificial Neural Network
 *
 * @author Emre Sen
 * @date Dec 26, 2018
 * @contact maemresen07@gmail.com
 */
public class ANN implements Serializable {

	private static final long serialVersionUID = -299482035708790407L;
	private static final Logger LOGGER = LoggerFactory.getLogger(ANN.class);

	private DataSet trainDataSet;
	private Layer inputLayer;

	/**/
	private double alpha;
	private Double lambda;
	private int numOfIterations;
	private boolean untilConverge;
	private boolean normalization;

	private int epochSize;
	private int numOfEpochs;

	private int iteration;

	public ANN(DataSet dataSet, double alpha, Double lambda, int numOfIterations, boolean untilConverge, boolean normalization, int epochSize,
			int numOfInputs, int numOfClasses, List<Integer> hiddenLayersNumOfUnitList) {

		this.trainDataSet = dataSet;

		// init layers
		this.inputLayer = new Layer("Input Layer", numOfInputs, Layer.LayerType.INPUT);
		Layer accLayer = inputLayer;
		int i = 0;
		for (Integer hiddenLayerNumOfInput : hiddenLayersNumOfUnitList) {
			Layer hiddenLayer = new Layer("Hidden Layer " + (++i), hiddenLayerNumOfInput, Layer.LayerType.HIDDEN);
			accLayer.setNextLayer(hiddenLayer);
			accLayer = hiddenLayer;
		}
		Layer outputLayer = new Layer("Output Layer", numOfClasses, Layer.LayerType.OUTPUT);
		accLayer.setNextLayer(outputLayer);

		// init params
		this.alpha = alpha;
		this.lambda = lambda;
		this.numOfIterations = numOfIterations;
		this.untilConverge = untilConverge;
		this.normalization = normalization;

		if (epochSize > trainDataSet.getSampleSize()) {
			LOGGER.warn("Number of Epochs could not be larger than sample size" + trainDataSet.getSampleSize());
			this.epochSize = trainDataSet.getSampleSize();
			this.numOfEpochs = 1;
		} else {
			this.epochSize = epochSize;
			this.numOfEpochs = trainDataSet.getSampleSize() / epochSize;
		}

		LOGGER.info("Number of Epochs is " + numOfEpochs);
		this.iteration = 0;
	}

	/**/
	public DataSet getTrainDataSet() {
		return trainDataSet;
	}

	public Layer getInputLayer() {
		return inputLayer;
	}

	public double getAlpha() {
		return alpha;
	}

	public Double getLambda() {
		return lambda;
	}

	public int getNumOfIterations() {
		return numOfIterations;
	}

	public boolean isUntilConverge() {
		return untilConverge;
	}

	public int getEpochSize() {
		return epochSize;
	}

	public int getNumOfEpochs() {
		return numOfEpochs;
	}

	public int getIteration() {
		return iteration;
	}

	public boolean isNormalization() {
		return normalization;
	}

	/**/
	public void setAlpha(double alpha) {
		this.alpha = alpha;
	}

	public void setLambda(Double lambda) {
		this.lambda = lambda;
	}

	public void setNumOfIterations(int numOfIterations) {
		this.numOfIterations = numOfIterations;
	}

	public void setUntilConverge(boolean untilConverge) {
		this.untilConverge = untilConverge;
	}

	/*
	 * ------------------------------- Prediction -------------------------------
	 */

	/**
	 * Forwarding inputs through the Artificial Neural ANN layers
	 *
	 * @param input each row corresponds sample each column corresponds features of
	 *              one sample
	 * @return predicted result for given samples
	 */
	private RealMatrix forward(RealMatrix input) {
		RealMatrix acc = (normalization) ? trainDataSet.applyMeanNormalization(input) : input;
		RealMatrix biased = MatrixHelper.addBiasTermToMatrix(acc);
		return inputLayer.forward(biased);
	}

	/**
	 * Predicting values of given samples
	 *
	 * @param input each row corresponds sample, each column corresponds features of
	 *              one sample
	 * @return predicted result for given samples
	 */
	public RealMatrix predict(RealMatrix input) {
		return forward(input);
	}

	/*
	 * ------------------------------- Training -------------------------------
	 */

	/**
	 * Calculating cost for given dataset with current theta values (weights)
	 *
	 * @return the cost of the given dataset
	 */
	private double cost() {

		RealMatrix input = trainDataSet.getFeaturesMatrix();
		RealMatrix values = trainDataSet.getValuesMatrix();
		double m = trainDataSet.getSampleSize();

		LOGGER.trace("Start Calculating Hypothesis");
		RealMatrix hypothesis = forward(input);
		LOGGER.trace("Finish Calculating Hypothesis");

		LOGGER.trace("Start Calculating Cost Matrix");
		RealMatrix leftPart = MatrixHelper.dotProduct(values, MatrixHelper.log(hypothesis));
		RealMatrix rightPart = MatrixHelper.dotProduct(values.scalarMultiply(-1).scalarAdd(1),
				MatrixHelper.log(hypothesis.scalarMultiply(-1).scalarAdd(1)));
		RealMatrix costMatrix = leftPart.add(rightPart);
		double costScalar = ((-1.0) / m);
		double cost = costScalar * MatrixHelper.sum(costMatrix);
		LOGGER.trace("Finish Calculating Cost Matrix");

		if (lambda != null) {
			LOGGER.trace("Start Adding Regularization Term to Cost");
			double regularizationScalar = ((-1.0 * lambda) / (2.0 * m));
			double regularizationTerm = inputLayer.sumThetaSquares();
			cost += regularizationScalar * regularizationTerm;
			LOGGER.trace("Finish Adding Regularization Term to Cost");
		}
		return cost;
	}

	private void backward(RealMatrix values) {
		inputLayer.updateThetas(values, values.getRowDimension(), alpha, lambda);
	}

	private boolean gradientDecent(double initialCost) {

		double cost = initialCost;
		LOGGER.trace("Start Apply BackPropagation");
		RealMatrix features = trainDataSet.getFeaturesMatrix();
		RealMatrix values = trainDataSet.getValuesMatrix();

		for (int e = 0; e < numOfEpochs; e++) {
			int startRowIndex = e * epochSize;
			int endRowIndex = startRowIndex + epochSize - 1;
			if (endRowIndex >= trainDataSet.getSampleSize()) {
				endRowIndex = trainDataSet.getSampleSize() - 1;
			}

			// fp
			RealMatrix epochFeatures = features.getSubMatrix(startRowIndex, endRowIndex, 0,
					features.getColumnDimension() - 1);
			forward(epochFeatures);

			// bp
			RealMatrix epochValues = values.getSubMatrix(startRowIndex, endRowIndex, 0,
					values.getColumnDimension() - 1);
			backward(epochValues);

			LOGGER.trace("Start Calculating Cost");
			double newCost = cost();
			String info2 = String.format("Iteration %s Epoch %s Cost : %s", iteration, (e + 1), newCost);
			LOGGER.info(info2);
			LOGGER.trace("Finish Calculating Cost");

			boolean converged = (Math.abs((cost - newCost)) < 0.001);
			if (untilConverge && converged) {
				return true;
			}
			cost = newCost;
		}
		LOGGER.trace("Finish Apply BackPropagation");
		return false;
	}

	/**
	 * Train ANN on given DataSet with given learning parameters
	 */
	public void learn() {

		// TODO: depends on epoch

		LOGGER.trace("Start Calculating Initial Cost");
		double initialCost = cost();
		LOGGER.info("Init Cost : " + initialCost);
		LOGGER.trace("Finish Calculating Initial Cost");

		LOGGER.info("Start Gradient Decent");
		int i = 0;
		while (true) {
			iteration++;
			i++;
			LOGGER.trace("Start Iteration " + iteration);

			LOGGER.trace("Finish Iteration " + iteration);
			boolean converged = gradientDecent(initialCost);
			if (untilConverge && converged) {
				break;
			}
			if (!untilConverge && (i >= numOfIterations)) {
				break;
			}
		}
		LOGGER.info("Gradient Step Completed with " + iteration + " Iteration");
		LOGGER.trace("Finish Gradient Decent");

		LOGGER.info("Final Results");
		LOGGER.info(" Initial Cost before G.D. : " + initialCost);
		LOGGER.info("Minimized Cost after G.D. : " + cost());
	}

	/*
	 * ------------------------------- Accuracy Test -------------------------------
	 */
	public void accuracy(DataSet testDataSet) {

		RealMatrix features = testDataSet.getFeaturesMatrix();

		double[] estimations = MatrixHelper.max(testDataSet.getValuesMatrix()).getColumn(0);
		double[][] results = MatrixHelper.max(predict(features)).getData();

		double[] classes = testDataSet.getClasses();
		double[] corrects = new double[testDataSet.getNumOfClasses()];

		for (int i = 0; i < estimations.length; i++) {
			double estimated = estimations[i];
			double predicted = results[i][0];
			double probability = results[i][1] * 100;
			boolean correct = estimated == predicted;
			for (int c = 0; c < corrects.length; c++) {
				if (classes[c] == estimated) {
					if (correct) {
						corrects[c]++;
					}
					break;
				}
			}
			LOGGER.debug(String.format("Sample (%s) : (%s)", i, correct));
			LOGGER.debug(String.format(" Estimated : %s", estimated));
			LOGGER.debug(String.format(" Predicted : %s with %.0f%% probability", predicted, probability));
		}

		LOGGER.info("Layers");
		Layer layer = inputLayer;
		do {
			LOGGER.info("   Units : " + layer.getNumOfUnits());
			layer = layer.getNextLayer();
		} while (layer != null);

		LOGGER.info("Parameters");
		LOGGER.info("   Iterations 	  : " + iteration);
		LOGGER.info("   Alpha      	  : " + alpha);
		LOGGER.info("   Lambda     	  : " + lambda);
		LOGGER.info("   Normalization : " + normalization);
		LOGGER.info("   Results");
		LOGGER.info("Num of Instances of the Classes : " + Arrays.toString(testDataSet.getNumOfClassInstances()));
		LOGGER.info("Correct Predicted Class Counts  : " + Arrays.toString(corrects));

		LOGGER.info("Statistics");
		double totalAccuracy = 0;
		for (int c = 0; c < corrects.length; c++) {
			double accuracy = (corrects[c] / testDataSet.getNumOfClassInstances()[c]);
			totalAccuracy += corrects[c] / testDataSet.getSampleSize();
			LOGGER.info(String.format("Class %s Accuracy : %.2f%%", (c + 1), accuracy * 100));
			LOGGER.info(String.format("   Number of Samples : %.0f", testDataSet.getNumOfClassInstances()[c]));
			LOGGER.info(String.format("   Correct Predicted : %.0f", corrects[c]));
		}

		LOGGER.info(String.format("   Total Accuracy   : %.2f%%", totalAccuracy * 100));

	}

	public void print() {
		Layer layer = inputLayer;
		do {
			layer.print();
			layer = layer.getNextLayer();
		} while (layer != null);
	}

}
