package com.maemresen.ml.hw1.util.loader;

import java.io.*;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * @author Emre Sen
 * @date Dec 9, 2018
 * @contact maemresen07@gmail.com
 */
public class ObjectSerializer {

    /**
     * Serialize given object into given file.
     *
     * @param file   file that serialized object will be stored
     * @param object object will be serialized
     * @return serialized object saved successfully
     */
    public static boolean serialize(String file, Object object) {
        try (FileOutputStream fos = new FileOutputStream(file);
             ObjectOutputStream oos = new ObjectOutputStream(fos)) {
            oos.writeObject(object);
            return true;
        } catch (IOException ex) {
            Logger.getLogger(ObjectSerializer.class.getName()).log(Level.SEVERE, null, ex);
        }
        return false;
    }

    /**
     * Deserialize given object from given file.
     *
     * @param file file that contains serialized object will be loaded
     * @return optional deserialized object if it can
     */
    public static Optional<Object> deserialize(String file) {
        // read object from file
        try (FileInputStream fis = new FileInputStream(file);
             ObjectInputStream ois = new ObjectInputStream(fis)) {
            return Optional.of(ois.readObject());
        } catch (ClassNotFoundException | IOException ex) {
            Logger.getLogger(ObjectSerializer.class.getName()).log(Level.SEVERE, null, ex);
        }
        return Optional.empty();
    }

}
