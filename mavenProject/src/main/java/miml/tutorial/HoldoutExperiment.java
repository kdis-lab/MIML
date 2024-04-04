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
		System.out.println("Example1:");
		System.out.println("\tjava -jar HoldoutExperiment -f data" + File.separator + "miml_birds.arff -x data"
				+ File.separator + "miml_birds.xml -r output" + File.separator + "miml_birds_report.csv");
		System.out.println("Example2:");
		System.out.println("\tjava -jar HoldoutExperiment -x data" + File.separator + "miml_birds.xml -t data"
				+ File.separator + "miml_birds_random_80train.arff" + File.separator + " -y data" + File.separator
				+ "miml_birds_random_20test.arff -r output" + File.separator + "miml_birds_report.csv");
		System.exit(-1);
	}

	public static void main(String[] args) throws Exception {

		// Example 1 => -x data/miml_birds.xml -f data/miml_birds.arff -r
		// results/report.csv
		// Example 2 => -x data/miml_birds.xml -t data/miml_birds_random_80train.arff -y
		// data/miml_birds_random_20test.arff -r results/report.csv

		// MIML report
		String reportFileName = Utils.getOption("r", args);
		BaseMIMLReport report = new BaseMIMLReport(null, reportFileName, false, false, false);

		// Loads classifier
		System.out.println("Loading MIMLkNN classifier...");
		MIMLkNN mimlknn = new MIMLkNN(new MaximalHausdorff());

		System.out.println("Loading datasets...");

		String xmlFileName = Utils.getOption("x", args);
		String arffFileName = Utils.getOption("f", args);

		if (arffFileName.equals("")) {
			// Holdout evaluation providing train and test partitions
			String arffFileNameTrain = Utils.getOption("t", args);
			String arffFileNameTest = Utils.getOption("y", args);
			MIMLInstances mimlDataSetTrain = new MIMLInstances(arffFileNameTrain, xmlFileName);
			MIMLInstances mimlDataSetTest = new MIMLInstances(arffFileNameTest, xmlFileName);

			// Evaluator with train and test partitions
			EvaluatorHoldout holdoutTT = new EvaluatorHoldout(mimlDataSetTrain, mimlDataSetTest);

			System.out.println("\n");

			System.out.println("-Example using MIMLkNN with train/test datasets:\n");
			holdoutTT.runExperiment(mimlknn);
			System.out.println(report.toString(holdoutTT) + "\n\n");
			report.saveReport(report.toCSV(holdoutTT));

		}

		else {

			// Holdout evaluation providing percentageTrain
			MIMLInstances mimlDataSet = new MIMLInstances(arffFileName, xmlFileName);

			// Evaluator with a percentage of train
			double percentageTrain = 80;
			EvaluatorHoldout holdoutSplit = new EvaluatorHoldout(mimlDataSet, percentageTrain);

			System.out.println("-Example using MIMLkNN with percentage of train:\\n");
			holdoutSplit.runExperiment(mimlknn);
			System.out.println(report.toString(holdoutSplit) + "\n\n");
			report.saveReport(report.toCSV(holdoutSplit));
		}

		System.out.println("The program has finished.");

	}
}
