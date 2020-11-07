package com.maemresen.ml.hw1.util.helper;

/**
 * @author Emre Sen
 * @date Dec 26, 2018
 * @contact maemresen07@gmail.com
 */
public class MathHelper {

    public static double log(double x) {
        return Math.log(x);
    }

    public static double sigmoid(double z) {
        return (1 / (1 + Math.pow(Math.E, (-1 * z))));
    }


}
