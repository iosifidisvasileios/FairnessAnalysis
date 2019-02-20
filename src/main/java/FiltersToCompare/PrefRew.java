package FiltersToCompare;

import org.apache.log4j.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Enumeration;

import static java.lang.StrictMath.abs;

public class PrefRew {
    protected double favPos=0,favNeg=0,savPos=0,savNeg=0;
    final static Logger logger = Logger.getLogger(PrefRew.class);

    public static int sa_Index;
    private Instances weightedInstances;
    double [][]spList; // list of instances with SA=dep   class=dc
    double [][]snList;// list of instances with SA=dep   class=ndc
    double [][]fpList;// list of instances with SA=fav   class=dc
    double [][]fnList;   // list of instances with SA=fav   class=ndc
    double [][]restList;   // list of instances with SA other than Dep and fav  i.e. when SA has more than two values
    //    intitializatin of sensitive varialbles with the corresponding values of The Discrimination class
    int dc=1,ndc=0;
    public static String sa_Deprived, sa_Favored;  //sa= sensitive attribute   sa_deprived=deprived community
    //sa_favored=favord community of sa


    public PrefRew(Instances training, double r, int saINDEX, String saDeprived) throws Exception {
        sa_Index = saINDEX;
        if (saDeprived.equals(" Female")) {
            sa_Deprived  = "' Female'";
        }else if(saDeprived.equals(" Minorities")){
            sa_Deprived  = "' Minorities'";
        }else {
            sa_Deprived  = saDeprived;
        }
//        logger.info("given range = " + r);
        this.weightedInstances = this.calculateWeightedDataset(training, abs(r));
    }

    public Instances getWeightedInstances() {
        return weightedInstances;
    }

    private Instances calculateWeightedDataset(Instances data, double r) throws Exception {

        final Instances weightedIntances = new Instances(data);
        ArrayList<Double> weights = weightCalculation(data);
//        System.out.println(weights);
        double topicWeights;
        double weightsforinstance;
        ArrayList<double[][]> prob=this.orderProbability(weightedIntances);

        for (int j=0 ; j<=weights.size()-1 ; j++) {
            int k=1;
            topicWeights=weights.get(j);

            for (double[] row : prob.get(j)) {
                double index = row[0];
                int n=prob.get(j).length;
                if (r!=0) {
                    weightsforinstance = topicWeights * (1 - r + 2 * (n - k + 1) * r / (n - 1)) + 1;
                }else{
                    weightsforinstance = topicWeights * (1 - r + 2 * (n - k + 1) * r / (n - 1)) ;
                }
                weightedIntances.instance((int)index).setWeight(weightsforinstance);
//                logger.info(weightedIntances.instance((int)index).weight());
                k++;
            }
        }

        return weightedIntances;
    }
//
//    private static void evaluateModel(Classifier classifier, Instances train, Instances test, String method) throws Exception {
//        logger.info("-------------- " + method + " -----------------\n");
//        System.out.println(Discrimination.discCalculation(train)+" discrimination on training set");
//
//        Instances predicted=new Instances(train,0);
//
//        classifier.buildClassifier(train);
//        final Evaluation eval = new Evaluation(train);
//        eval.evaluateModel(classifier, test);
//        logger.info("Accuracy =" + eval.pctCorrect());
//
//        for(Instance inst : test) {
//            inst.setClassValue(classifier.classifyInstance(inst));
//            predicted.add(inst);
//        }
//        logger.info(Discrimination.discCalculation(predicted)+" Discrimination sta Predictions");
//    }

    public static Classifier classifier=new NaiveBayes();//NaiveBayesSimple

    int sp=0,sn=0,fp=0,fn=0,restCount=0;

    public ArrayList<double[][]> orderProbability (Instances instances)throws Exception {

        classifier.buildClassifier(instances);

        spList = new double[instances.numInstances()][2];
        snList = new double[instances.numInstances()][2];
        fpList = new double[instances.numInstances()][2];
        fnList = new double[instances.numInstances()][2];

        double[] prob = new double[instances.numClasses()]; //array to store probabilty distribution
        ArrayList<double[][]> retn = new ArrayList<double[][]>();
        Enumeration enumInsts = instances.enumerateInstances();
        int classValue = 0, instIndex = 0;
        sp = sn = fp = fn = restCount = 0;
        String saValue;
        while (enumInsts.hasMoreElements()) {
            Instance instance = (Instance) enumInsts.nextElement();
//            System.out.println(instance);

            saValue = instance.toString(sa_Index);
            classValue = (int) instance.classValue();
            if (saValue.equals(sa_Deprived) && classValue == dc) {
                spList[sp][0] = instIndex;
                prob = classifier.distributionForInstance(instance);
                spList[sp++][1] = prob[dc] * 100;
            } else if (saValue.equals(sa_Deprived) && classValue == ndc) {
                snList[sn][0] = instIndex;
                prob = classifier.distributionForInstance(instance);
                snList[sn++][1] = prob[dc] * 100;
            } else if (!saValue.equals(sa_Deprived) && classValue == dc) {
                fpList[fp][0] = instIndex;
                prob = classifier.distributionForInstance(instance);
                fpList[fp++][1] = prob[dc] * 100;
            } else if (!saValue.equals(sa_Deprived) && classValue == ndc) {
                fnList[fn][0] = instIndex;
//                System.out.println(instance);
                prob = classifier.distributionForInstance(instance);
                fnList[fn++][1] = prob[dc] * 100;
            }
            instIndex++;
        }

        spList = MyMassaging.sorting(spList, sp, 2); //women positive
        snList = MyMassaging.sorting(snList, sn, 1); //women negative
        fpList = MyMassaging.sorting(fpList, fp, 2); //men positive
        fnList = MyMassaging.sorting(fnList, fn, 1);  //men negative
//        restList = Massaging.sorting(restList, restCount, 1);
//        retn.add([spList],[snList],[fpList],[fnList],[restList]);

        retn.add(spList);
        retn.add( snList);
        retn.add(fpList);
        retn.add(fnList);

        return  retn;
    }

    public ArrayList<Double> weightCalculation(Instances instances)throws Exception{
        spList=new double[instances.numInstances()][2];
        snList=new double[instances.numInstances()][2];
        fpList=new double[instances.numInstances()][2];
        fnList=new double[instances.numInstances()][2];
        restList=new double[instances.numInstances()][2];
        ArrayList weights=new ArrayList<Double>();
        Enumeration enumInsts = instances.enumerateInstances();
        int classValue;
        sp=sn=fp=fn=restCount=0;
        String saValue;
        while (enumInsts.hasMoreElements()) {
            Instance instance = (Instance) enumInsts.nextElement();
            saValue=instance.toString(sa_Index);
            classValue=(int)instance.classValue();

            if(saValue.equals(sa_Deprived) && classValue==dc ){
                sp++;
            }
            else if(saValue.equals(sa_Deprived) && classValue==ndc){
                sn++;
            }
            else if(!saValue.equals(sa_Deprived) && classValue==dc){
                fp++;
            }
            else if(!saValue.equals(sa_Deprived) && classValue==ndc){
                fn++;
            }
            else{

                restCount++;
            }
        }
        double total=instances.numInstances();
//        System.out.println("SavPos=: "+savPos+"  SavNeg =: "+savNeg+"   saPos =: "+favPos+"  saNeg =: "+favNeg);
//        System.out.println("DG=: "+sp+"  DR=: "+sn+"   FG=: "+fp+"  FR=: "+fn);
        savPos=((sp+sn)/total)*((sp+fp)/sp);
        savNeg=((sp+sn)/total)*((sn+fn)/sn);
        favPos=((fp+fn)/total)*((sp+fp)/fp);
        favNeg=((fp+fn)/total)*((sn+fn)/fn);
        weights.add(savPos);
        weights.add(savNeg);
        weights.add(favPos);
        weights.add(favNeg);
//        System.out.println("Weights: SavPos=: "+savPos+"  SavNeg =: "+savNeg+"   saPos =: "+favPos+"  saNeg =: "+favNeg);
//        System.out.println("SP=: "+sp+"  SN =: "+sn+"   Fp =: "+fp+"  fn =: "+fn);
        return weights;
    }

}