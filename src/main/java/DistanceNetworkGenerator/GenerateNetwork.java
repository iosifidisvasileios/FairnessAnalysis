package DistanceNetworkGenerator;

import FiltersToCompare.MyMassagingFilter;
import FiltersToCompare.MyPrefrentialSamplingFilter;
import org.apache.log4j.Logger;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.filters.Filter;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static java.lang.System.exit;

;

/**
 * Class for building and using a Classyfing without Discriminating classfierRanker.
 * Numeric attributes are modelled by a normal distribution. For more
 * Discrimination has been calculated by taking the difference of confidence of SA values for decired class(DC)
 * Discrimination is removed by using the reweighing technique of CND
 * Author Toon Calders, Faisal Kamran
 */
public class GenerateNetwork {


    private static String protectedValueName;
    private static int protectedValueIndex;
    private static String targetClass;
    private static String otherClass;
    private static String outfile;
    private final static Logger log = Logger.getLogger(GenerateNetwork.class.getName());

    public static void main(String [] argv) throws Exception {

//
        final String parameters = "adult-gender";
        final String classifier = "NB";
//        final String preprocessing= "Original";
//        final String preprocessing= "massaging";
        final String preprocessing= "sampling";

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
        } else if (parameters.equals("propublica")) {
            reader = new BufferedReader(new FileReader("Datasets/compass_zafar.arff"));
            protectedValueName = "1";
            protectedValueIndex = 4;
            targetClass = "1";
            otherClass = "-1";
        } else {
            exit(1);
        }

        outfile = parameters + "-" + classifier + "/";


        int nearestNeighbor = 10;

        Instances data = new Instances(reader);
        reader.close();


        if (preprocessing.equals("sampling")) {
            Filter.filterFile(new MyPrefrentialSamplingFilter(protectedValueName, protectedValueIndex),
                    new String[]{"-i", "Datasets/adult.arff","-o", "Datasets/temp_preprocess.arff", "-c", "last"});
            reader = new BufferedReader(new FileReader("Datasets/temp_preprocess.arff"));
            data = new Instances(reader);
            reader.close();
        }else if (preprocessing.equals("massaging")) {
            Filter.filterFile(new MyMassagingFilter(protectedValueName, protectedValueIndex),
                    new String[]{"-i", "Datasets/adult.arff", "-o", "Datasets/temp_preprocess.arff", "-c", "last"});
            reader = new BufferedReader(new FileReader("Datasets/temp_preprocess.arff"));
            data = new Instances(reader);
            reader.close();
        }

        BufferedWriter writerNode = new BufferedWriter(new FileWriter("Datasets/DistanceNetwork_" + parameters +"_" + preprocessing +"_node.csv"));
        writerNode.write("Id,Category\n");

        BufferedWriter writerEdge = new BufferedWriter(new FileWriter("Datasets/DistanceNetwork_" + parameters +"_" + preprocessing +"_edge.csv"));
        writerEdge.write("Source,Target,Weight\n");

        EuclideanDistance m1_distanceFunction = new EuclideanDistance(data);

        if (parameters.startsWith("adult"))
            m1_distanceFunction.setAttributeIndices("1-13");


        ArrayList<Double> bufferList = new ArrayList<>();

        for(int i = 0; i < data.size(); i++){
            writerNode.write(i +"," + data.get(i).stringValue(protectedValueIndex) + "_"+ data.get(i).stringValue(data.numAttributes()- 1) + "\n");

            for(int j = 0; j < data.size(); j++) {
                if(j==i)
                    continue;
                double dist = m1_distanceFunction.distance(data.get(i), data.get(j));
                bufferList.add(dist);
            }

            int[] indeces = getLowest(bufferList, nearestNeighbor);

            for (int index = 0; index < nearestNeighbor; index++ ){
                writerEdge.write(i + "," + indeces[index] + "," + bufferList.get(indeces[index])+ "\n");
            }


            bufferList.clear();
        }
        writerEdge.close();
        writerNode.close();

    }



    private static int[] getLowest(ArrayList<Double> list, int howMany) {
        int[] indices = new int[howMany];

        ArrayList<Double> copy = new ArrayList<>(list);
        Collections.sort(copy);

        List<Double> lowest = new ArrayList<>();

        for (int i = 0; i < howMany; i++) {
            lowest.add(copy.get(i));
        }

        int indicesIndex = 0;
        for (int d = 0; d < list.size(); d++) {
            if (lowest.contains(list.get(d))) {
                indices[indicesIndex] = d;
                indicesIndex++;
            }
            if (indicesIndex == howMany)
                break;
        }

        return indices;
    }
}
