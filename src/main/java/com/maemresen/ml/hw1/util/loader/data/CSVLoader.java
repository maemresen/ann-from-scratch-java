package com.maemresen.ml.hw1.util.loader.data;

import com.maemresen.ml.hw1.util.ann.DataSet;
import com.maemresen.ml.hw1.util.helper.ArrayHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringTokenizer;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * @author Emre Sen
 * @date Jan 02, 2019
 * @contact maemresen07@gmail.com
 */
public class CSVLoader {

    private static final Logger LOGGER = LoggerFactory.getLogger(DataLoader.class);

    public static DataSet loadDataSet(String dataFileName) throws IOException {
        LOGGER.trace("Start Loading data form file");
        double[][] data = processInputFile(dataFileName);
        LOGGER.trace("Finish Loading data from file");
        return new DataSet(data);
    }

    private static double[][] processInputFile(String dataFileName) throws IOException {


        try (InputStream bufferedIn = new BufferedInputStream(DataLoader.class.getResourceAsStream("/datasets/" + dataFileName));
             InputStreamReader inR = new InputStreamReader(bufferedIn);
             BufferedReader br = new BufferedReader(inR)) {
            return br.lines().skip(1).map(mapToItem).toArray(double[][]::new);
        }
    }


    private static Function<String, double[]> mapToItem = (line) -> {
        String[] p = line.split(",");// a CSV has comma separated lines
        double[] result = new double[p.length];
        result[p.length - 1] = Double.parseDouble(p[0]);
        for (int i = 1; i < p.length; i++) {
            result[i - 1] = Double.parseDouble(p[i]);
        }
        return result;
    };

}
