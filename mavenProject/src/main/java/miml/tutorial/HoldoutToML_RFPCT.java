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

import miml.classifiers.miml.mimlTOml.MIMLClassifierToML;
import miml.classifiers.ml.RFPCT;
import miml.core.ConfigParameters;
import miml.data.MIMLInstances;
import miml.evaluation.EvaluatorHoldoutClus;
import miml.report.BaseMIMLReport;
import miml.transformation.mimlTOml.ArithmeticTransformation;
import weka.core.Utils;

/**
 * 
 * Class implementing an example of using holdout with train/test dataset and a
 * toML classifier with RFPCT as base classifier.
 * 
 * @author Eva Gibaja
 * @version 20230306
 *
 */
public class HoldoutToML_RFPCT {
	public static void main(String[] args) throws Exception {

		// Example=> -x data/miml_birds.xml -t data/miml_birds_random_80train.arff -y
		// data/miml_birds_random_20test.arff -r results/report.csv

		String reportFileName = Utils.getOption("r", args);
		String xmlFileName = Utils.getOption("x", args);
		String arffFileNameTrain = Utils.getOption("t", args);
		String arffFileNameTest = Utils.getOption("y", args);

		System.out.println("Example using Arithmetic transformation and RFPCT with train/test datasets.\n ");

		// MIML report
		BaseMIMLReport report = new BaseMIMLReport(null, reportFileName, false, false, false);
		ConfigParameters.setAlgorithmName("ToML_AT_RFPCT"); // value for Algorithm column in the csv report
		ConfigParameters.setConfigFileName("Java app"); // value for ConfigurationFile column in the csv report. As the
		// experiment does not use any configuration file, "Java app"
		// has been set to avoid an empty column
		ConfigParameters.setDataFileName(arffFileNameTrain); // value for Dataset column in the csv report

		// Loads classifier
		String clusWorkingDir = new String("clusFolder");
		String clusDatasetName = new String("datasetName");
		System.out.println("Loading classifier...");
		MIMLClassifierToML mimltoml = new MIMLClassifierToML(new RFPCT(clusWorkingDir, clusDatasetName, 10, 1),
				new ArithmeticTransformation());

		// Loads datasets
		System.out.println("Loading datasets...");
		MIMLInstances mimlDataSetTrain = new MIMLInstances(arffFileNameTrain, xmlFileName);
		MIMLInstances mimlDataSetTest = new MIMLInstances(arffFileNameTest, xmlFileName);

		// Evaluator with train and test partitions
		EvaluatorHoldoutClus holdoutTT = new EvaluatorHoldoutClus(mimlDataSetTrain, mimlDataSetTest, clusWorkingDir,
				clusDatasetName);

		System.out.println("\n");

		holdoutTT.runExperiment(mimltoml);

		System.out.println("\nThe clus library execution has finished.\n");

		System.out.println(
				"NOTE that that RFPCT calls clus library that performs, in a single call, train and test steps. Therefore:");
		System.out.println("\t1. Train time got by miml library is not relevant.");
		System.out.println(
				"\t2. Test time got by miml libraryr really computes the train and test time required by the call to clus library.\n\n");
		System.out.println(report.toString(holdoutTT) + "\n\n");
		report.saveReport(report.toCSV(holdoutTT));

		System.out.println("The program has finished.");

	}
}
