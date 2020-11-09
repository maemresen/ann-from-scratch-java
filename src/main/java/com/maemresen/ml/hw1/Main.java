package com.maemresen.ml.hw1;

import java.io.IOException;

import org.apache.commons.math3.linear.RealMatrix;

import com.maemresen.ml.hw1.examples.part1.Part1BalancedDatasetTrainer;
import com.maemresen.ml.hw1.examples.part1.Part1NodeSizeTrainer;
import com.maemresen.ml.hw1.examples.part1.Part1NormalizationTrainer;
import com.maemresen.ml.hw1.examples.part1.Part1RegularizationTrainer;
import com.maemresen.ml.hw1.examples.part1.Part1WeightUpdateTrainer;
import com.maemresen.ml.hw1.examples.part2.Part2NodeSizeTrainer;
import com.maemresen.ml.hw1.util.ann.DataSet;
import com.maemresen.ml.hw1.util.loader.ModelLoader;
import com.maemresen.ml.hw1.util.loader.data.CSVLoader;
import com.maemresen.ml.hw1.util.loader.data.DataLoader;

public class Main {

    public static void main(String[] args) throws IOException {
//        part1TrainExamples();
//        part2TrainExamples();
    }

    private static void part1TrainExamples() throws IOException {

        DataSet trainDataSet = DataLoader.loadDataSet("ann-train.data");
        DataSet testDataSet = DataLoader.loadDataSet("ann-test.data");

        /* For different values of the hidden unit number */
        String trainAnn3NodeFileName = Part1NodeSizeTrainer.trainAnn3Node();
        String trainAnn11NodeFileName = Part1NodeSizeTrainer.trainAnn11Node();
        String trainAnn22NodeFileName = Part1NodeSizeTrainer.trainAnn22Node();
        String trainAnn44NodeFileName = Part1NodeSizeTrainer.trainAnn44Node();
        String trainAnn88NodeFileName = Part1NodeSizeTrainer.trainAnn88Node();

        /* Balanced / Unbalanced */
        String trainAnnBalancedTrainDatasetFileName = Part1BalancedDatasetTrainer.trainAnnBalancedTrainDataset();
        String trainAnnUnalancedTrainDatasetFileName = Part1BalancedDatasetTrainer.trainAnnUnalancedTrainDataset();

        /* Regularization */
        String trainAnnWithouthRegularizationFileName = Part1RegularizationTrainer.trainAnnWithouthRegularization();
        String trainAnnWithRegularizationTest1FileName = Part1RegularizationTrainer.trainAnnWithRegularizationTest1();
        String trainAnnWithRegularizationTest2FileName = Part1RegularizationTrainer.trainAnnWithRegularizationTest2();

        /* Weight Update Approach */
        String trainAnnWithBatchFileName = Part1WeightUpdateTrainer.trainAnnWithBatch();
        String trainAnnWithMiniBatchFileName = Part1WeightUpdateTrainer.trainAnnWithMiniBatch();
        String trainAnnWithStochaticFileName = Part1WeightUpdateTrainer.trainAnnWithStochatic();

        /* Normalization */
        String trainAnnWithNormalizationFileName = Part1NormalizationTrainer.trainAnnWithNormalization();
        String trainAnnWithoutNormalizationFileName = Part1NormalizationTrainer.trainAnnWithoutNormalization();

//        test(trainAnn3NodeFileName, trainDataSet);
//        test(trainAnn3NodeFileName, testDataSet);
    }

    private static void part2TrainExamples() throws IOException {
        DataSet mnistTestDataSet = CSVLoader.loadDataSet("mnist_test.csv");
        RealMatrix s = CSVLoader.loadDataSet("mnist_train.csv").getDataMatrix();
        DataSet mnistTrainDataSet = new DataSet(s.getSubMatrix(0, 30000, 0, s.getColumnDimension() - 1).getData());

        String trainAnnl1_5_l2_10 = Part2NodeSizeTrainer.trainAnnl1_5_l2_10();

//        test(trainAnnl1_5_l2_10, mnistTrainDataSet);
//        test(trainAnnl1_5_l2_10, mnistTestDataSet);
    }

    private static void test(String filename, DataSet dataSet) {
        ModelLoader.loadModel(filename).ifPresent(ann -> {
            ann.accuracy(dataSet);
        });
    }

}
