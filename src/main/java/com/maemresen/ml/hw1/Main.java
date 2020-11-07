package com.maemresen.ml.hw1;

import java.io.IOException;

import org.apache.commons.math3.linear.RealMatrix;

import com.maemresen.ml.hw1.part1.Part1BalancedDatasetTest;
import com.maemresen.ml.hw1.part1.Part1NodeSizeTest;
import com.maemresen.ml.hw1.part1.Part1NormalizationTest;
import com.maemresen.ml.hw1.part1.Part1RegularizationTest;
import com.maemresen.ml.hw1.part1.Part1WeightUpdateTest;
import com.maemresen.ml.hw1.part2.Part2NodeSizeTest;
import com.maemresen.ml.hw1.util.ann.DataSet;
import com.maemresen.ml.hw1.util.loader.ModelLoader;
import com.maemresen.ml.hw1.util.loader.data.CSVLoader;
import com.maemresen.ml.hw1.util.loader.data.DataLoader;

public class Main {

	public static void main(String[] args) throws IOException {

//		part1Tests();
//		part2Tests();

//		DataSet annTrainDataSet = DataLoader.loadDataSet("ann-train.data");
//		DataSet annTestDataSet = DataLoader.loadDataSet("ann-test.data");

		DataSet mnistTestDataSet = CSVLoader.loadDataSet("mnist_test.csv");
		RealMatrix s = CSVLoader.loadDataSet("mnist_train.csv").getDataMatrix();
		DataSet mnistTrainDataSet = new DataSet(s.getSubMatrix(0, 30000, 0, s.getColumnDimension() - 1).getData());

		String file = "network_models/part2/network-l1_20-l2_10.dat";
		test(file, mnistTrainDataSet);
		test(file, mnistTestDataSet);
	}

	private static void part1Tests() throws IOException {

		DataSet trainDataSet = DataLoader.loadDataSet("ann-train.data");
		DataSet testDataSet = DataLoader.loadDataSet("ann-test.data");

		/* For different values of the hidden unit number */
		Part1NodeSizeTest.trainAnn3Node();
		Part1NodeSizeTest.trainAnn11Node();
		Part1NodeSizeTest.trainAnn22Node();
		Part1NodeSizeTest.trainAnn44Node();
		Part1NodeSizeTest.trainAnn88Node();

		/* Balanced / Unbalanced */
		Part1BalancedDatasetTest.trainAnnBalancedTrainDataset();
		Part1BalancedDatasetTest.trainAnnUnalancedTrainDataset();

		/* Regularization */
		Part1RegularizationTest.trainAnnWithouthRegularization();
		Part1RegularizationTest.trainAnnWithRegularizationTest1();
		Part1RegularizationTest.trainAnnWithRegularizationTest2();

		/* Weight Update Approach */
		Part1WeightUpdateTest.trainAnnWithBatch();
		Part1WeightUpdateTest.trainAnnWithMiniBatch();
		Part1WeightUpdateTest.trainAnnWithStochatic();

		/* Normalization */
		Part1NormalizationTest.trainAnnWithNormalization();
		Part1NormalizationTest.trainAnnWithoutNormalization();

	}

	private static void part2Tests() throws IOException {

		Part2NodeSizeTest.trainAnnl1_5_l2_10();
//		Part2NodeSizeTest.trainAnnl1_10_l2_10();
//		Part2NodeSizeTest.trainAnnl1_20_l2_10();
	}

	private static void test(String filename, DataSet dataSet) {
		ModelLoader.loadModel(filename).ifPresent(ann -> {
			ann.accuracy(dataSet);
		});
	}

}
