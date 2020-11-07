package com.maemresen.ml.hw1.part1;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.maemresen.ml.hw1.util.ann.ANN;
import com.maemresen.ml.hw1.util.ann.DataSet;
import com.maemresen.ml.hw1.util.loader.ModelLoader;
import com.maemresen.ml.hw1.util.loader.data.DataBalancers;
import com.maemresen.ml.hw1.util.loader.data.DataLoader;

/**
 *
 * @author Emre Sen
 * @date Jan 01, 2018
 * @contact maemresen07@gmail.com
 */

public class Part1NormalizationTest {

	private static final Logger LOGGER = LoggerFactory.getLogger(Part1NormalizationTest.class);

	private static String trainAnn(boolean n) throws IOException {

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
		boolean normalization = n;
		int epochSize = trainDataSet.getSampleSize();
		int numOfInputs = trainDataSet.getFeatureSize() + 1;
		int numOfClasses = trainDataSet.getNumOfClasses();
		List<Integer> hiddenLayersNumOfUnitList = Collections.singletonList(11);

		ANN ann = new ANN(dataset,
				/* Learning Parameters */
				alpha, lambda, numOfIterations, untilConverge, normalization, epochSize,
				/* Layer Informations */
				numOfInputs, numOfClasses, hiddenLayersNumOfUnitList);
		LOGGER.trace("Finish Creating an Artificial Neural Network");

		LOGGER.info("Start Learning");
		ann.learn();
		LOGGER.trace("Finish Learning");
		String filename = "network_models/part1/05_normalization/network-"
				+ ((normalization) ? "normalized" : "not-normalized") + ".dat";
		ModelLoader.save(ann, filename);
		return filename;
	}

	public static String trainAnnWithNormalization() throws IOException {
		return trainAnn(true);
	}

	public static String trainAnnWithoutNormalization() throws IOException {
		return trainAnn(false);
	}
}
