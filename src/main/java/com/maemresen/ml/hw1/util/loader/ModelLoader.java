package com.maemresen.ml.hw1.util.loader;

import com.maemresen.ml.hw1.util.ann.ANN;
import org.apache.commons.io.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Optional;
import java.util.Scanner;

/**
 * @author Emre Sen
 * @date Dec 9, 2018
 * @contact maemresen07@gmail.com
 */
public class ModelLoader {

    private static final Logger LOGGER = LoggerFactory.getLogger(ModelLoader.class);

    /**
     * Save model to file
     *
     * @return save operation success or not
     */
    public static boolean save(ANN model) {
        try (Scanner scanner = new Scanner(System.in)) {
            LOGGER.info("Would you like to save the network model? (y/n) : ");
            if (!scanner.nextLine().equals("y")) {
                return false;
            }
            LOGGER.info("Enter the file name : ");

            String filename = scanner.nextLine();
            return save(model, filename);
        }
    }

    /**
     * Save model to file
     *
     * @return save operation success or not
     */
    public static boolean save(ANN model, String filename) {
        try (Scanner scanner = new Scanner(System.in)) {
            FileUtils.forceMkdirParent(new File(filename));
            try (FileOutputStream fos = new FileOutputStream(filename);
                 ObjectOutputStream oos = new ObjectOutputStream(fos)) {
                // write model to file
                oos.writeObject(model);
            }
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Load object from file
     *
     * @return loaded object
     */
    public static Optional<ANN> loadModel() {
        try (Scanner scanner = new Scanner(System.in)) {
            LOGGER.info("Enter the file name : ");
            String filename = scanner.nextLine();
            return loadModel(filename);
        }
    }

    /**
     * Load object from file
     * @param filename contains object
     * @return loaded object
     */
    public static Optional<ANN> loadModel(String filename) {
        return ObjectSerializer.deserialize(filename).map(o -> (ANN) o);
    }
}
