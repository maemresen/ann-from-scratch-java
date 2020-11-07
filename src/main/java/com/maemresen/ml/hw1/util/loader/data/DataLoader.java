package com.maemresen.ml.hw1.util.loader.data;

import com.maemresen.ml.hw1.util.ann.DataSet;
import com.maemresen.ml.hw1.util.helper.ArrayHelper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.function.Function;

/**
 * @author Emre Sen
 * @date Dec 22, 2018
 * @contact maemresen07@gmail.com
 */
public class DataLoader {

    private static final Logger LOGGER = LoggerFactory.getLogger(DataLoader.class);

    /**
     * @param dataFileName ..
     * @param balancers    ..
     * @return ...
     * @throws IOException ..
     */
    @SafeVarargs
    public static DataSet loadDataSet(String dataFileName, Function<DataSet, DataSet>... balancers) throws IOException {
        LOGGER.trace("Start Loading data form file");
        double[][] data = loadData(dataFileName);
        LOGGER.trace("Finish Loading data from file");
        DataSet dataSet = new DataSet(data);
        if (balancers == null) {
            return dataSet;
        }
        DataSet result = dataSet;
        if (balancers.length != 0) {
            LOGGER.trace("Start Balancing Data with balancers");
        }
        for (Function<DataSet, DataSet> balancer : balancers) {
            if (balancer == null) {
                continue;
            }
            LOGGER.trace("Start Balance Data");
            result = balancer.apply(result);
            LOGGER.trace("Finish Balance Data");
        }
        LOGGER.trace("Finish Balancing Data with balancers");
        return result;
    }

    private static double[][] loadData(String dataFileName) throws IOException {

        try (InputStream bufferedIn = new BufferedInputStream(DataLoader.class.getResourceAsStream("/datasets/" + dataFileName));
             InputStreamReader inR = new InputStreamReader(bufferedIn);
             BufferedReader buf = new BufferedReader(inR)) {

            List<List<String>> samples = new ArrayList<>();
            String line;

            while ((line = buf.readLine()) != null) {
                StringTokenizer st = new StringTokenizer(line, " ");
                List<String> features = new ArrayList<>();
                while (st.hasMoreTokens()) {
                    features.add(st.nextToken());
                }
                samples.add(features);
            }
            return ArrayHelper.listTo2DDoubleArray(samples);
        }
    }


}
