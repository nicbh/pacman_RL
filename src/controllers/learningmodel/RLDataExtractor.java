/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package controllers.learningmodel;

import tools.*;
import core.game.Observation;
import core.game.StateObservation;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Observable;

import ontology.Types;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

import javax.swing.text.Position;

/**
 * @author yuy
 */
public class RLDataExtractor {
    public static Instances s_datasetHeader = datasetHeader();
    public FileWriter filewriter;

    public RLDataExtractor(String filename) throws Exception {

        filewriter = new FileWriter(filename + ".arff");
        filewriter.write(s_datasetHeader.toString());
        /*
                // ARFF File header
        filewriter.write("@RELATION AliensData\n");
        // Each row denotes the feature attribute
        // In this demo, the features have four dimensions.
        filewriter.write("@ATTRIBUTE gameScore  NUMERIC\n");
        filewriter.write("@ATTRIBUTE avatarSpeed  NUMERIC\n");
        filewriter.write("@ATTRIBUTE avatarHealthPoints NUMERIC\n");
        filewriter.write("@ATTRIBUTE avatarType NUMERIC\n");
        // objects
        for(int y=0; y<14; y++)
            for(int x=0; x<32; x++)
                filewriter.write("@ATTRIBUTE object_at_position_x=" + x + "_y=" + y + " NUMERIC\n");
        // The last row of the ARFF header stands for the classes
        filewriter.write("@ATTRIBUTE Class {0,1,2}\n");
        // The data will recorded in the following.
        filewriter.write("@Data\n");*/

    }

    public static Instance makeInstance(double[] features, int eff, int action, double reward) {
        features[674] = eff;
        features[875] = action;
        features[876] = reward;
        Instance ins = new Instance(1, features);
        ins.setDataset(s_datasetHeader);
        return ins;
    }

    private static double disMan(Vector2d a1, Vector2d a2) {
        return Math.abs(a1.x - a2.x) + Math.abs(a1.y - a2.y);
    }

    public static double[] featureExtract(StateObservation obs) {

        double[] feature = new double[877];  // 868 + 6 + 1 + 1(action) + 1(Q)

        // 448 locations
        int[][] map = new int[28][31];
        // Extract features
        LinkedList<Observation> allobj = new LinkedList<>();
        if (obs.getImmovablePositions() != null)
            for (ArrayList<Observation> l : obs.getImmovablePositions()) allobj.addAll(l);
        if (obs.getMovablePositions() != null)
            for (ArrayList<Observation> l : obs.getMovablePositions()) allobj.addAll(l);
        if (obs.getNPCPositions() != null)
            for (ArrayList<Observation> l : obs.getNPCPositions()) allobj.addAll(l);

        for (Observation o : allobj) {
            Vector2d p = o.position;
            int x = (int) (p.x / 20); //squre size is 20 for pacman
            int y = (int) (p.y / 20);
            map[x][y] = o.itype;
        }
        Vector2d pos = obs.getAvatarPosition();
        map[(int) (pos.x / 20)][(int) (pos.y / 20)] = obs.getAvatarType();
        for (int y = 0; y < 31; y++)
            for (int x = 0; x < 28; x++)
                feature[y * 28 + x] = map[x][y];

        double[] distance = {1000, 1000, 1000, 1000};
        if (obs.getNPCPositions() != null) {
            for (ArrayList<Observation> l : obs.getNPCPositions()) {
                Observation obser = l.get(0);
                switch (obser.itype) {
                    case 14:
                    case 15:
                        distance[0] = disMan(obser.position, pos);
                        break;
                    case 17:
                    case 18:
                        distance[1] = disMan(obser.position, pos);
                        break;
                    case 20:
                    case 21:
                        distance[2] = disMan(obser.position, pos);
                        break;
                    case 23:
                    case 24:
                        distance[3] = disMan(obser.position, pos);
                        break;
                }
            }
        }

        // 4 states
        feature[868] = obs.getGameTick();
//        feature[869] = obs.getAvatarSpeed();
//        feature[870] = obs.getAvatarHealthPoints();
        feature[869] = obs.getAvatarType();
        feature[870] = distance[0];
        feature[871] = distance[1];
        feature[872] = distance[2];
        feature[873] = distance[3];

        return feature;
    }

    public static Instances datasetHeader() {

        if (s_datasetHeader != null)
            return s_datasetHeader;

        FastVector attInfo = new FastVector();
        // 448 locations
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 31; x++) {
                Attribute att = new Attribute("object_at_position_x=" + x + "_y=" + y);
                attInfo.addElement(att);
            }
        }
        Attribute att = new Attribute("GameTick");
        attInfo.addElement(att);
//        att = new Attribute("AvatarSpeed");
//        attInfo.addElement(att);
//        att = new Attribute("AvatarHealthPoints");
//        attInfo.addElement(att);
        att = new Attribute("AvatarType");
        attInfo.addElement(att);
        att = new Attribute("Distance1");
        attInfo.addElement(att);
        att = new Attribute("Distance2");
        attInfo.addElement(att);
        att = new Attribute("Distance3");
        attInfo.addElement(att);
        att = new Attribute("Distance4");
        attInfo.addElement(att);
        //action
        FastVector actions = new FastVector();
        actions.addElement("0");
        actions.addElement("1");
        att = new Attribute("EffectAction", actions);
        attInfo.addElement(att);

        actions = new FastVector();
        actions.addElement("0");
        actions.addElement("1");
        actions.addElement("2");
        actions.addElement("3");
        att = new Attribute("actions", actions);
        attInfo.addElement(att);
        // Q value
        att = new Attribute("Qvalue");
        attInfo.addElement(att);

        Instances instances = new Instances("PacmanQdata", attInfo, 0);
        instances.setClassIndex(instances.numAttributes() - 1);

        return instances;
    }

}
