/*    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

package miml.tutorial;

import java.io.File;

import miml.classifiers.miml.lazy.MIMLkNN;
import miml.core.distance.MaximalHausdorff;
import miml.data.MIMLInstances;
import miml.evaluation.EvaluatorHoldout;
import miml.report.BaseMIMLReport;
import weka.core.Utils;

/**
 * 
 * Class implementing an example of using holdout with train/test dataset and a
 * single dataset applying percentage split.
 * 
 * @author Alvaro A. Belmonte
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20190525
 *
 */
public class HoldoutExperiment {

	/** Shows the help on command line. */
	public static void showUse() {
		System.out.println("Program parameters:");
		System.out.println("\t-f arffPathFile Name -> path of arff source file.");
		System.out.println("\t-x xmlPathFileName -> path of xml file.");
		System.out.println("\t-t arffPathFileTrain Name -> path of arff train file.");
		System.out.println("\t-y arffPathFileTest -> path of arff test file.");
		System.out.println("\t-r reportPathFileName -> path of report file.");
		System.out.println("Example:");
		System.out.println("\tjava -jar HoldoutExperiment -f data" + File.separator + "miml_birds.arff -x data"
				+ File.separator + "miml_birds.xml -t data" + File.separator + "miml_birds_random_80train.arff"
				+ File.separator + " -y data" + File.separator + "miml_birds_random_20test.arff -r output"
				+ File.separator + "miml_birds_report.csv");
		System.exit(-1);
	}

	public static void main(String[] args) throws Exception {

		//-f  data/birds.arff -x data/birds.xml -t data/birds.arff -y data/birds.arff -r results/report.csv
		
		String arffFileName = Utils.getOption("f", args);
		String xmlFileName = Utils.getOption("x", args);
		String arffFileNameTrain = Utils.getOption("t", args);
		String arffFileNameTest = Utils.getOption("y", args);
		String reportFileName = Utils.getOption("r", args);

		
		// Loads the dataset
		System.out.println("Loading datasets...");
		MIMLInstances mimlDataSet = new MIMLInstances(arffFileName, xmlFileName);
		MIMLInstances mimlDataSetTrain = new MIMLInstances(arffFileNameTrain, xmlFileName);
		MIMLInstances mimlDataSetTest = new MIMLInstances(arffFileNameTest, xmlFileName);

		// MIML report
		BaseMIMLReport report = new BaseMIMLReport(null, reportFileName, false, false, false);

		// Cross-validation evaluator
		EvaluatorHoldout holdoutTT = new EvaluatorHoldout(mimlDataSetTrain, mimlDataSetTest);
		EvaluatorHoldout holdoutSplit = new EvaluatorHoldout(mimlDataSet, 80);

		// Load first classifier
		System.out.println("Loading MIMLkNN classifier...");
		MIMLkNN mimlknn = new MIMLkNN(new MaximalHausdorff(mimlDataSet));

		System.out.println("\n");

		System.out.println("-Example using MIMLkNN with train/test datasets:\n");
		holdoutTT.runExperiment(mimlknn);
		System.out.println(report.toString(holdoutTT) + "\n\n");
		report.saveReport(report.toCSV(holdoutTT));

		System.out.println("-Example using MIMLkNN with dataset split:\\n");
		holdoutSplit.runExperiment(mimlknn);
		System.out.println(report.toString(holdoutSplit) + "\n\n");
		report.saveReport(report.toCSV(holdoutTT));

		System.out.println("The program has finished.");
	}
}
