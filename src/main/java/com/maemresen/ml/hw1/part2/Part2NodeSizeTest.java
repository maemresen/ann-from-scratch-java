package com.maemresen.ml.hw1.part2;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.maemresen.ml.hw1.util.ann.ANN;
import com.maemresen.ml.hw1.util.ann.DataSet;
import com.maemresen.ml.hw1.util.loader.ModelLoader;
import com.maemresen.ml.hw1.util.loader.data.CSVLoader;

/**
 *
 * @author Emre Sen
 * @date Jan 01, 2018
 * @contact maemresen07@gmail.com
 */
public class Part2NodeSizeTest {

	private static final Logger LOGGER = LoggerFactory.getLogger(Part2NodeSizeTest.class);

	private static String trainAnn(int hl1NodeNumber, int hl2NodeNumber) throws IOException {

		LOGGER.info("Start Loading Train Dataset");
		DataSet trainDataSet = CSVLoader.loadDataSet("mnist_train.csv");
		LOGGER.trace("Finish Loading Dataset");

		LOGGER.trace("Start Creating an Artificial Neural Network");

		DataSet dataset = trainDataSet;
		double alpha = 1.5;
		Double lambda = null;
		int numOfIterations = 10;
		boolean untilConverge = false;
		boolean normalization = true;
		int epochSize = 128;
		int numOfInputs = trainDataSet.getFeatureSize() + 1;
		int numOfClasses = trainDataSet.getNumOfClasses();
		List<Integer> hiddenLayersNumOfUnitList = Arrays.asList(hl1NodeNumber, hl2NodeNumber);

		ANN ann = new ANN(dataset,
				/* Learning Parameters */
				alpha, lambda, numOfIterations, untilConverge, normalization, epochSize,
				/* Layer Informations */
				numOfInputs, numOfClasses, hiddenLayersNumOfUnitList);
		LOGGER.trace("Finish Creating an Artificial Neural Network");

		LOGGER.info("Start Learning");
		ann.learn();
		LOGGER.trace("Finish Learning");
		String filename = "network_models/part2/network-l1_" + hl1NodeNumber + "-l2_" + hl2NodeNumber + ".dat";
		ModelLoader.save(ann, filename);
		return filename;
	}

	public static String trainAnnl1_5_l2_10() throws IOException {
		return trainAnn(5, 10);
	}

	public static String trainAnnl1_10_l2_10() throws IOException {
		return trainAnn(10, 10);
	}

	public static String trainAnnl1_20_l2_10() throws IOException {
		return trainAnn(20, 10);
	}

}
