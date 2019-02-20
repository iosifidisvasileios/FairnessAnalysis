package FiltersToCompare;

import org.apache.log4j.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;

/**
 * Created by iosifidis on 20.03.18.
 */
public class LibraryForMyFunctions {

    public LibraryForMyFunctions() {
    }
    private final static Logger log = Logger.getLogger(LibraryForMyFunctions.class.getName());

    public static HashMap<String, Double> evaluate_pre_existing_methodsWeighted(String classifier,
                                                             Instances training,
                                                             String testingDir,
                                                             int protectedValueIndex,
                                                             String protectedValueName,
                                                             String targetClass,
                                                             String otherClass,
                                                             String method) throws Exception {

        BufferedReader reader = new BufferedReader(new FileReader(testingDir));
        final Instances testing  = new Instances(reader);
        reader.close();

        training.setClassIndex(training.numAttributes() - 1);
        testing.setClassIndex(training.numAttributes() - 1);
//        log.info("---------------------" + method + "------------------------");


        if (classifier.equals("NB")) {
            NaiveBayes nb = new NaiveBayes();
            // EvaluateClassifier(nb, training, testing, "Naive Bayes");
            return exportDiscrimination(nb, "Naive Bayes", testing, protectedValueIndex, protectedValueName, targetClass, otherClass, training);

        } else if (classifier.equals("J48")) {
            J48 j48 = new J48();
            // EvaluateClassifier(j48, training, testing, "J48");
            return exportDiscrimination(j48, "J48", testing, protectedValueIndex, protectedValueName, targetClass, otherClass, training);

        } else if (classifier.equals("Ada")) {
            AdaBoostM1 adaboost = new AdaBoostM1();
            // EvaluateClassifier(adaboost, training, testing, "Adaboost");
            return exportDiscrimination(adaboost, "Adaboost", testing, protectedValueIndex, protectedValueName, targetClass, otherClass, training);

        } else  {
            RandomForest randomForest = new RandomForest();
            // EvaluateClassifier(randomForest, training, testing, "randomForest");
            return exportDiscrimination(randomForest, "randomForest", testing, protectedValueIndex, protectedValueName, targetClass, otherClass, training);
        }

    }





    public static HashMap<String, Double> evaluate_pre_existing_methods(String classifier,
                                                     String trainingDir,
                                                     String testingDir,
                                                     int protectedValueIndex,
                                                     String protectedValueName,
                                                     String targetClass,
                                                     String otherClass,
                                                     String method) throws Exception {

        BufferedReader reader = new BufferedReader(new FileReader(trainingDir));
        final Instances training = new Instances(reader);
        reader.close();

        reader = new BufferedReader(new FileReader(testingDir));
        final Instances testing  = new Instances(reader);
        reader.close();

        training.setClassIndex(training.numAttributes() - 1);
        testing.setClassIndex(training.numAttributes() - 1);
//        log.info("---------------------" + method + "------------------------");


        if (classifier.equals("NB")) {
            NaiveBayes nb = new NaiveBayes();
            // EvaluateClassifier(nb, training, testing, "Naive Bayes");
            return exportDiscrimination(nb, "Naive Bayes", testing, protectedValueIndex, protectedValueName, targetClass, otherClass, training);

        } else if (classifier.equals("J48")) {
            J48 j48 = new J48();
            // EvaluateClassifier(j48, training, testing, "J48");
            return exportDiscrimination(j48, "J48", testing, protectedValueIndex, protectedValueName, targetClass, otherClass, training);

        } else if (classifier.equals("Ada")) {
            AdaBoostM1 adaboost = new AdaBoostM1();
            // EvaluateClassifier(adaboost, training, testing, "Adaboost");
            return exportDiscrimination(adaboost, "Adaboost", testing, protectedValueIndex, protectedValueName, targetClass, otherClass, training);

        } else  {
            RandomForest randomForest = new RandomForest();
            // EvaluateClassifier(randomForest, training, testing, "randomForest");
            return exportDiscrimination(randomForest, "randomForest", testing, protectedValueIndex, protectedValueName, targetClass, otherClass, training);
        }


    }


    public static HashMap<String, Double> exportDiscrimination(Classifier classifier,
                                                               String model,
                                                               Instances testing,
                                                               int protectedValueIndex,
                                                               String protectedValueName,
                                                               String targetClass,
                                                               String otherClass, Instances training) throws Exception {
        HashMap<String, Double> map = new HashMap<String, Double>();
        String nonProtectedValueName = "";
        classifier.buildClassifier(training);
        final Evaluation eval = new Evaluation(training);
        eval.evaluateModel(classifier, testing);

        log.info(model + " Accuracy = " + eval.pctCorrect());
//        log.info(model + " Au-ROC  = " + eval.weightedAreaUnderROC() * 100);
//        log.info(model + " Au-PRC  = " + eval.weightedAreaUnderPRC() * 100);

        map.put("Accuracy", eval.pctCorrect());
        map.put("ROC", eval.weightedAreaUnderROC() * 100);
        map.put("PRC", eval.weightedAreaUnderPRC() * 100);



        Instances TestingPredictions = new Instances(testing);

        double tp_male = 0;
        double tn_male = 0;
        double tp_female = 0;
        double tn_female = 0;
        double fp_male = 0;
        double fn_male = 0;
        double fp_female = 0;
        double fn_female = 0;
        double female = 0;
        double male = 0;

        double CorrectFemale = 0;
        double CorrectMale = 0;

        double femalePositiveProb = 0;
        double malePositiveProb = 0;

        double femaleNegativeProb = 0;
        double maleNegativeProb = 0;

        double mismatch = 0;
        for(Instance ins: TestingPredictions){
//            log.info(ins.stringValue(protectedValueIndex));
            if (!ins.stringValue(protectedValueIndex).equals(protectedValueName)) {
                nonProtectedValueName = ins.stringValue(protectedValueIndex);
                break;
            }
        }

        for(Instance ins: TestingPredictions){

            double label = classifier.classifyInstance(ins);
            // WOMEN
            if (ins.stringValue(protectedValueIndex).equals(protectedValueName)) {
                female += 1;
                // correctly classified
                if (label == ins.classValue()) {
                    CorrectFemale +=1;
                    // on target class (true negatives)
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        femalePositiveProb +=Math.max(classifier.distributionForInstance(ins)[0],classifier.distributionForInstance(ins)[1]);
                        tp_female++;

                    }else if(ins.stringValue(ins.classIndex()).equals(otherClass)){
                        tn_female++;
                        femaleNegativeProb +=Math.max(classifier.distributionForInstance(ins)[0],classifier.distributionForInstance(ins)[1]);

                    }
                }else{
                    // error has been made on TN so it's FP
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        fn_female++;
                        femaleNegativeProb +=Math.max(classifier.distributionForInstance(ins)[0],classifier.distributionForInstance(ins)[1]);

                    }else if (ins.stringValue(ins.classIndex()).equals(otherClass)){
                        femalePositiveProb +=Math.max(classifier.distributionForInstance(ins)[0],classifier.distributionForInstance(ins)[1]);

                        // error has been made on TP so it's FN
                        fp_female++;

                    }
                }
            }else{
                male +=1;
                // correctly classified
                if (label == ins.classValue()) {
//                    log.info(ins.attribute(protectedValueIndex));

                    CorrectMale +=1;
                    // on target class (true negatives)
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        tp_male++;
                        malePositiveProb +=Math.max(classifier.distributionForInstance(ins)[0],classifier.distributionForInstance(ins)[1]);

                    }else if(ins.stringValue(ins.classIndex()).equals(otherClass)){
                        tn_male++;
                        maleNegativeProb +=Math.max(classifier.distributionForInstance(ins)[0],classifier.distributionForInstance(ins)[1]);

                    }
                }else{
                    if (ins.stringValue(ins.classIndex()).equals(targetClass)) {
                        // error has been made on TP so it's FN
                        maleNegativeProb +=Math.max(classifier.distributionForInstance(ins)[0],classifier.distributionForInstance(ins)[1]);
                        fn_male++;

                    }else if(ins.stringValue(ins.classIndex()).equals(otherClass)){
                        // error has been made on TN so it's FP
                        malePositiveProb +=Math.max(classifier.distributionForInstance(ins)[0],classifier.distributionForInstance(ins)[1]);
                        fp_male++;

                    }
                }
            }
//            if (ins.stringValue(protectedValueIndex).equals(protectedValueName)) {
//                ins.setValue(protectedValueIndex, nonProtectedValueName);
//            } else{
//                ins.setValue(protectedValueIndex, protectedValueName);
//            }
//
//            double labelModified = classifier.classifyInstance(ins);
//
//            if (label != labelModified){
//                mismatch +=1;
//            }

        }

        log.info("fp_male = "   + fp_male);
        log.info("fn_male = "   + fn_male);
        log.info("fp_female = " + fp_female);
        log.info("fn_female = " + fn_female);

        log.info("tp_male = "   + tp_male);
        log.info("tn_male = "   + tn_male);
        log.info("tp_female = " + tp_female);
        log.info("tn_female = " + tn_female);

        double statisticalParity = (fp_male + tp_male) / male - (fp_female + tp_female) / female;
        double predictiveEquality = (tn_male)/(tn_male + fp_male) - (tn_female)/(tn_female + fp_female);
        double accuracyEquality = CorrectMale/male - CorrectFemale/female;
        double equalOpportunity = (tp_male)/(tp_male + fn_male) - (tp_female)/(tp_female + fn_female);
        double balancePos = malePositiveProb/(tp_male + fp_male) - femalePositiveProb/(tp_female + fp_female);
        double balanceNeg = maleNegativeProb/(tn_male + fn_male) - femaleNegativeProb/(tn_female + fn_female);
        double causalDisc = mismatch/ (double)TestingPredictions.size();

        log.info(model + " statisticalParity  = " + statisticalParity    * 100 );
//        log.info(model + " causalDisc         = " + causalDisc           * 100 );
//        log.info(model + " AccuracyEquality   = " + accuracyEquality     * 100 );
//        log.info(model + " equal opportunity  = " + equalOpportunity     * 100 );
//        log.info(model + " predictiveEquality = " + predictiveEquality   * 100 );
//        log.info(model + " balancePositive    = " + balancePos           * 100 );
//        log.info(model + " balanceNegative    = " + balanceNeg           * 100 );

        map.put("statisticalParity", statisticalParity*100);
        map.put("causalDisc", causalDisc*100);
        map.put("accuracyEquality", accuracyEquality*100);
        map.put("eqOp", equalOpportunity*100);
        map.put("predictiveEquality", predictiveEquality*100);
        map.put("balancePositive", balancePos*100);
        map.put("balanceNegative", balanceNeg*100);


        return map;
    }


}
