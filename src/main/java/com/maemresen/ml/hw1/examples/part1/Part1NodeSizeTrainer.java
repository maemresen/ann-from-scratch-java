package com.maemresen.ml.hw1.examples.part1;

import com.maemresen.ml.hw1.util.ann.ANN;
import com.maemresen.ml.hw1.util.ann.DataSet;
import com.maemresen.ml.hw1.util.loader.ModelLoader;
import com.maemresen.ml.hw1.util.loader.data.DataBalancers;
import com.maemresen.ml.hw1.util.loader.data.DataLoader;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 *
 * @author Emre Sen
 * @date Jan 01, 2018
 * @contact maemresen07@gmail.com
 */
public class Part1NodeSizeTrainer {

	private static final Logger LOGGER = LoggerFactory.getLogger(Part1NodeSizeTrainer.class);

	private static String trainAnn(int hlNodeNumber) throws IOException {

		LOGGER.info("Start Loading Train Dataset");
		DataSet trainDataSet = DataLoader.loadDataSet("ann-train.data",
				/**/
				DataBalancers::limitToAvg,
				/**/
				DataBalancers::duplicateMin,
				/**/
				DataBalancers::duplicateMin,
				/**/
				DataBalancers::duplicateMin,
				/**/
				DataBalancers::duplicateAboveAvg);
		LOGGER.trace("Finish Loading Dataset");

		LOGGER.trace("Start Creating an Artificial Neural Network");

		DataSet dataset = trainDataSet;
		double alpha = 1.5;
		Double lambda = null;
		int numOfIterations = 30000;
		boolean untilConverge = false;
		boolean normalization = true;
		int epochSize = trainDataSet.getSampleSize();
		int numOfInputs = trainDataSet.getFeatureSize() + 1;
		int numOfClasses = trainDataSet.getNumOfClasses();
		List<Integer> hiddenLayersNumOfUnitList = Collections.singletonList(hlNodeNumber);

		ANN ann = new ANN(dataset,
				/* Learning Parameters */
				alpha, lambda, numOfIterations, untilConverge, normalization, epochSize,
				/* Layer Informations */
				numOfInputs, numOfClasses, hiddenLayersNumOfUnitList);
		LOGGER.trace("Finish Creating an Artificial Neural Network");

		LOGGER.info("Start Learning");
		ann.learn();
		LOGGER.trace("Finish Learning");
		String filename = "network_models/part1/01_nodesize/network-n" + hlNodeNumber + ".dat";
		ModelLoader.save(ann, filename);
		return filename;
	}

	public static String trainAnn3Node() throws IOException {
		return trainAnn(3);
	}

	public static String trainAnn11Node() throws IOException {
		return trainAnn(11);
	}

	public static String trainAnn22Node() throws IOException {
		return trainAnn(22);
	}

	public static String trainAnn44Node() throws IOException {
		return trainAnn(44);
	}

	public static String trainAnn88Node() throws IOException {
		return trainAnn(88);
	}

}
