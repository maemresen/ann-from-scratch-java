package com.maemresen.ml.hw1.examples.part1;

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
public class Part1BalancedDatasetTrainer {

	private static final Logger LOGGER = LoggerFactory.getLogger(Part1BalancedDatasetTrainer.class);

	private static String trainAnn(boolean balanced) throws IOException {

		LOGGER.info("Start Loading Train Dataset");
		DataSet trainDataSet;
		if (balanced) {
			trainDataSet = DataLoader.loadDataSet("ann-train.data",
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
		} else {
			trainDataSet = DataLoader.loadDataSet("ann-train.data");
		}

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
		String filename = "network_models/part1/02_balanced/network-" + ((balanced) ? "balanced" : "unbalanced")
				+ ".dat";
		ModelLoader.save(ann, filename);
		return filename;
	}

	public static String trainAnnBalancedTrainDataset() throws IOException {
		return trainAnn(true);
	}

	public static String trainAnnUnalancedTrainDataset() throws IOException {
		return trainAnn(false);
	}

}
