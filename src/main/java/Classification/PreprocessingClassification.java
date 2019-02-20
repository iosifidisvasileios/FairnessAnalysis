package Classification;


import FiltersToCompare.MyMassagingFilter;
import FiltersToCompare.MyPrefrentialSamplingFilter;
import FiltersToCompare.PrefRew;
import org.apache.commons.io.FileUtils;
import org.apache.log4j.Logger;
import org.tools4j.meanvar.MeanVarianceSlidingWindow;
import weka.core.Instances;
import weka.filters.Filter;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import static FiltersToCompare.LibraryForMyFunctions.evaluate_pre_existing_methods;
import static FiltersToCompare.LibraryForMyFunctions.evaluate_pre_existing_methodsWeighted;
import static java.lang.System.exit;


public class PreprocessingClassification {

    public static String protectedValueName;
    public static int protectedValueIndex;
    public static String targetClass;
    public static String otherClass;
    public static String outfile;
    private final static Logger log = Logger.getLogger(PreprocessingClassification.class.getName());

    public static void main(String [] argv) throws Exception {

//
        final String parameters = "adult-gender";
        final String classifier = "NB";
////
//        final String parameters = argv[0];
//        final String classifier = argv[1];


        final int folds = 10;
        final int iterations = 1;

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

        new File(outfile).mkdirs();

        ArrayList<HashMap<String, Double>> baseline = new ArrayList<HashMap<String, Double>>();
        ArrayList<HashMap<String, Double>> massaging = new ArrayList<HashMap<String, Double>>();
        ArrayList<HashMap<String, Double>> preferential = new ArrayList<HashMap<String, Double>>();
        ArrayList<HashMap<String, Double>> reweighting = new ArrayList<HashMap<String, Double>>();
        ArrayList<HashMap<String, Double>> SMOTE = new ArrayList<HashMap<String, Double>>();
        ArrayList<HashMap<String, Double>> OverSampling = new ArrayList<HashMap<String, Double>>();



        // how many times to run the experiment
        for (int k = 0; k < iterations; k++) {

            final Random rand = new Random((int) System.currentTimeMillis());   // create seeded number generator
            final Instances randData = new Instances(data);   // create copy of original data
            randData.randomize(rand);         // randomize data with number generator
            randData.setClassIndex(data.numAttributes() - 1);
            randData.stratify(folds);


            for (int n = 0; n < folds; n++) {
//                log.info("running fold = " + n);
                Instances train = randData.trainCV(folds, n);
                Instances test = randData.testCV(folds, n);

                WriteDataset(train, outfile + "temp_training_"  + "baselines.arff");
                WriteDataset(test, outfile + "temp_testing_"  + "baselines.arff");

                log.info("None");
                baseline.add(evaluate_pre_existing_methods(classifier,
                        outfile + "temp_training_"  + "baselines.arff",
                        outfile + "temp_testing_"  + "baselines.arff",
                        protectedValueIndex,
                        protectedValueName,
                        targetClass,
                        otherClass,
                        "Baseline"
                ));

                log.info("massaging");
                Filter.filterFile(new MyMassagingFilter(protectedValueName, protectedValueIndex),
                        new String[]{"-i", outfile + "temp_training_"  + "baselines.arff",
                                "-o", outfile + "temp_massaging_"  + "baselines.arff", "-c", "last"});
                massaging.add(evaluate_pre_existing_methods(classifier ,
                        outfile + "temp_massaging_"  + "baselines.arff",
                        outfile + "temp_testing_"  + "baselines.arff",
                        protectedValueIndex,
                        protectedValueName,
                        targetClass,
                        otherClass,
                        "Massaging"
                ));

                log.info("preferential");
                Filter.filterFile(new MyPrefrentialSamplingFilter(protectedValueName, protectedValueIndex),
                        new String[]{"-i", outfile + "temp_training_"  + "baselines.arff",
                                "-o", outfile + "temp_pref_"  + "baselines.arff", "-c", "last"});

                preferential.add(evaluate_pre_existing_methods(classifier ,
                        outfile + "temp_pref_" +  "baselines.arff",
                        outfile + "temp_testing_"  + "baselines.arff",
                        protectedValueIndex,
                        protectedValueName,
                        targetClass,
                        otherClass,
                        "Preferential Sampling"
                ));

                log.info("reweighting");
                PrefRew Reweighting1 = new PrefRew(train, 0, protectedValueIndex, protectedValueName);
                reweighting.add(evaluate_pre_existing_methodsWeighted(classifier ,
                        new Instances(Reweighting1.getWeightedInstances()),
                        outfile + "temp_testing_"  + "baselines.arff",
                        protectedValueIndex,
                        protectedValueName,
                        targetClass,
                        otherClass,
                        "Reweighting"
                ));

            }
        }

        generateStatistics("baseline", baseline,iterations *folds);
        generateStatistics("massaging", massaging,iterations *folds);
        generateStatistics("preferential", preferential,iterations *folds);
        generateStatistics("reweighting", reweighting,iterations *folds);

        FileUtils.deleteDirectory(new File(outfile));
    }



    public static void generateStatistics(String method, ArrayList<HashMap<String, Double>> mapper, int window) {

        final MeanVarianceSlidingWindow stdAcc = new MeanVarianceSlidingWindow(window);
        final MeanVarianceSlidingWindow stdROC = new MeanVarianceSlidingWindow(window);
        final MeanVarianceSlidingWindow stdPRC = new MeanVarianceSlidingWindow(window);
        final MeanVarianceSlidingWindow stdSP  = new MeanVarianceSlidingWindow(window);
        final MeanVarianceSlidingWindow stdAE  = new MeanVarianceSlidingWindow(window);
        final MeanVarianceSlidingWindow stdEO  = new MeanVarianceSlidingWindow(window);
        final MeanVarianceSlidingWindow stdPR  = new MeanVarianceSlidingWindow(window);
        final MeanVarianceSlidingWindow stdBP  = new MeanVarianceSlidingWindow(window);
        final MeanVarianceSlidingWindow stdBN  = new MeanVarianceSlidingWindow(window);
        final MeanVarianceSlidingWindow stdMM  = new MeanVarianceSlidingWindow(window);


        for(HashMap<String, Double> item : mapper){

            stdAcc.update(item.get("Accuracy"));
            stdROC.update(item.get("ROC"));
            stdPRC.update(item.get("PRC"));
            stdSP.update(item.get("statisticalParity"));
            stdAE.update(item.get("accuracyEquality"));
            stdEO.update(item.get("eqOp"));
            stdPR.update(item.get("predictiveEquality"));
            stdBP.update(item.get("balancePositive"));
            stdBN.update(item.get("balanceNegative"));
            stdMM.update(item.get("causalDisc"));
        }

        log.info(method + " Accuracy  = " + stdAcc.getMean() + ", " + stdAcc.getStdDev());
        log.info(method + " ROC  = " + stdROC.getMean() + ", " + stdROC.getStdDev());
        log.info(method + " PRC  = " + stdPRC.getMean() + ", " + stdPRC.getStdDev());
        log.info(method + " StatisticalParity  = " + stdSP.getMean() + ", " + stdSP.getStdDev());
        log.info(method + " AccuracyEquality   = " + stdAE.getMean() + ", " + stdAE.getStdDev());
        log.info(method + " EqOp  = " + stdEO.getMean() + ", " + stdEO.getStdDev());
        log.info(method + " PredictiveEquality = " + stdPR.getMean() + ", " + stdPR.getStdDev());
        log.info(method + " BalancePositive    = " + stdBP.getMean() + ", " + stdBP.getStdDev());
        log.info(method + " BalanceNegative    = " + stdBN.getMean() + ", " + stdBN.getStdDev());
        log.info(method + " causalDisc    = " + stdMM.getMean() + ", " + stdMM.getStdDev());

    }

    public static void WriteDataset(Instances data, String s) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(s));
        writer.write(data.toString());
        writer.flush();
        writer.close();

    }

}
