package Classification;


import org.apache.log4j.Logger;
import weka.attributeSelection.CorrelationAttributeEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.core.Instances;

import java.io.*;
import java.util.HashMap;

import static java.lang.System.exit;


public class CalculateSingleCorrelations {

    public static String protectedValueName;
    public static int protectedValueIndex;
    public static String targetClass;
    public static String otherClass;
    public static String outfile;
    private final static Logger log = Logger.getLogger(CalculateSingleCorrelations.class.getName());

    public static void main(String [] argv) throws Exception {

        final String parameters = "adult-gender";
        final String classifier = "NB";


        BufferedReader reader = null;
        if (parameters.equals("adult-gender")) {
            reader = new BufferedReader(new FileReader("Datasets/adult.arff"));
            protectedValueName = " Female";
            protectedValueIndex = 8;
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if (parameters.equals("adult-race")) {
            reader = new BufferedReader(new FileReader("Datasets/adult.arff"));
            protectedValueName = " Minorities";
            protectedValueIndex = 7;
            targetClass = " >50K";
            otherClass = " <=50K";
        } else if (parameters.equals("dutch")) {
            reader = new BufferedReader(new FileReader("Datasets/dutch.arff"));
            protectedValueName = "2"; // women ?
            protectedValueIndex = 0;
            targetClass = "2_1"; // high level ?
            otherClass = "5_4_9";
        } else if (parameters.equals("kdd")) {
            reader = new BufferedReader(new FileReader("Datasets/kdd.arff"));
            protectedValueName = "Female";
            protectedValueIndex = 12;
            targetClass = "1";
            otherClass = "0";
        } else {
            exit(1);
        }

        outfile = "Datasets/" + parameters + "-" + classifier + "/";

        final Instances data = new Instances(reader);
        reader.close();


        calculateCorrelations(data);

    }

    private static void calculateCorrelations(Instances data) throws Exception {
        data.setClassIndex(protectedValueIndex);

        GainRatioAttributeEval infoGain= new GainRatioAttributeEval();
        infoGain.buildEvaluator(data);
        HashMap<String,Double> attributeWeights = new HashMap<>();



        CorrelationAttributeEval correlationAttributeEval = new CorrelationAttributeEval();
        correlationAttributeEval.setOutputDetailedInfo(true);
        correlationAttributeEval.buildEvaluator(data);


        for (int i=0; i <data.numAttributes(); i++){
            attributeWeights.put(data.attribute(i).name(), 1 - Math.abs(infoGain.evaluateAttribute(i)));
            log.info("attribute " + data.attribute(i).name()  + ", gain ratio = " + infoGain.evaluateAttribute(i)*100
                    + ", pearson's correlation = " + correlationAttributeEval.evaluateAttribute(i)*100);
        }

        log.info(correlationAttributeEval.toString());

    }






}
