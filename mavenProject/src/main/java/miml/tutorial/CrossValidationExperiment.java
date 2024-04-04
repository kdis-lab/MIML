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

import miml.classifiers.miml.lazy.MIMLBRkNN;
import miml.classifiers.miml.lazy.MIMLDistanceFunction;
import miml.classifiers.miml.lazy.MIMLkNN;
import miml.classifiers.miml.meta.MIMLBagging;
import miml.classifiers.miml.mimlTOmi.MIMLClassifierToMI;
import miml.classifiers.miml.mimlTOmi.MIMLLabelPowerset;
import miml.classifiers.miml.mimlTOml.MIMLClassifierToML;
import miml.classifiers.miml.neural.MIMLRBF;
import miml.core.ConfigParameters;
import miml.core.distance.AverageHausdorff;
import miml.core.distance.MaximalHausdorff;
import miml.data.MIMLInstances;
import miml.evaluation.EvaluatorCV;
import miml.report.BaseMIMLReport;
import miml.transformation.mimlTOml.GeometricTransformation;
import miml.transformation.mimlTOml.MedoidTransformation;
import miml.transformation.mimlTOml.MinMaxTransformation;
import mulan.classifier.lazy.BRkNN;
import mulan.classifier.lazy.DMLkNN;
import mulan.classifier.transformation.LabelPowerset;
import weka.classifiers.mi.SimpleMI;
import weka.classifiers.trees.J48;
import weka.core.Utils;

/**
 * Class implementing an example of using cross-validation with different kinds
 * of classifier.
 * 
 * @author Alvaro A. Belmonte
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20190525
 *
 */
public class CrossValidationExperiment {

	/** Shows the help on command line. */
	public static void showUse() {
		System.out.println("Program parameters:");
		System.out.println("\t-f arffPathFile Name -> path of arff source file.");
		System.out.println("\t-x xmlPathFileName -> path of xml file.");
		System.out.println("\t-r reportPathFileName -> path of report file.");
		System.out.println("Example:");
		System.out.println("\tjava -jar CrossValidationExperiment -f data" + File.separator + "miml_birds.arff -x data"
				+ File.separator + "miml_birds.xml -r output" + File.separator + "miml_birds_report.csv");
		System.exit(-1);
	}

	public static void main(String[] args) throws Exception {

		// -f data/miml_birds.arff -x data/miml_birds.xml -r results/report.csv

		String arffFileName = Utils.getOption("f", args);
		String xmlFileName = Utils.getOption("x", args);
		String reportFileName = Utils.getOption("r", args);

		// Loads the dataset
		System.out.println("Loading the dataset...");
		MIMLInstances mimlDataSet = new MIMLInstances(arffFileName, xmlFileName);

		// MIML report
		boolean printStd = true, printMetricsPerLabel = false, printHeader = true;
		BaseMIMLReport report = new BaseMIMLReport(null, reportFileName, printStd, printMetricsPerLabel, printHeader);
		ConfigParameters.setConfigFileName("Java app"); // value for ConfigurationFile column in the csv report. As the
														// experiment does not use any configuration file, "Java app"
														// has been set to avoid an empty column
		ConfigParameters.setDataFileName(arffFileName); // value for Dataset column in the csv report

		// Cross-validation evaluator
		EvaluatorCV cv = new EvaluatorCV(mimlDataSet, 5);

		// Load classifiers
		System.out.println("Loading classifiers...");
		MIMLkNN mimlknn = new MIMLkNN(new MaximalHausdorff(mimlDataSet));
		MIMLClassifierToMI mimltomi = new MIMLClassifierToMI(new MIMLLabelPowerset(new SimpleMI()));
		MIMLClassifierToML mimltoml = new MIMLClassifierToML(new DMLkNN(), new GeometricTransformation());

		MedoidTransformation medoid = new MedoidTransformation();
		medoid.setPercentage(0.25);
		MIMLClassifierToML mimltoml_medoid = new MIMLClassifierToML(new BRkNN(), medoid);

		MIMLBRkNN miml_brknn = new MIMLBRkNN(new MIMLDistanceFunction(new AverageHausdorff()));
		MIMLBagging mimlbagging = new MIMLBagging(
				new MIMLClassifierToML(new LabelPowerset(new J48()), new MinMaxTransformation()), 10);
		MIMLRBF mimlrbf = new MIMLRBF(0.1, 0.6); // MW classifier

		System.out.println("\n");

		System.out.println("-First example cross-validation using MIMLkNN:\n");
		ConfigParameters.setAlgorithmName("MIMLkNN"); // value for Algorithm column in the csv report
		cv.runExperiment(mimlknn);
		System.out.println(report.toString(cv) + "\n\n");
		report.saveReport(report.toCSV(cv));
		report.setHeader(false); // Header is shown just for the first row

		System.out.println("-Second example cross-validation using MIMLtoMI transformation:\n");
		ConfigParameters.setAlgorithmName("toMI_LP_SimpleMI"); // value for Algorithm column in the csv report
		cv.runExperiment(mimltomi);
		System.out.println(report.toString(cv) + "\n\n");
		report.saveReport(report.toCSV(cv));

		System.out.println("-Third example cross-validation using MIMLtoML transformation:\n");
		ConfigParameters.setAlgorithmName("toML_GT_DMLkNN"); // value for Algorithm column in the csv report
		cv.runExperiment(mimltoml);
		System.out.println(report.toString(cv));
		report.saveReport(report.toCSV(cv));

		System.out.println("-Fourth example cross-validation using MIML_BRkNN:\n");
		ConfigParameters.setAlgorithmName("MIML_BRkNN_AveH"); // value for Algorithm column in the csv report
		cv.runExperiment(miml_brknn);
		System.out.println(report.toString(cv));
		report.saveReport(report.toCSV(cv));

		System.out.println("-Fifth example cross-validation using MIMLRBF:\n");
		ConfigParameters.setAlgorithmName("MIMLRBF"); // value for Algorithm column in the csv report
		cv.runExperiment(mimlrbf);
		System.out.println(report.toString(cv));
		report.saveReport(report.toCSV(cv));
		mimlrbf.dispose(); // Dispose of native MW resources

		System.out.println("-Sixth example cross-validation using MIMLBagging:\n");
		ConfigParameters.setAlgorithmName("MIMLBagging_MMT_LP_J48"); // value for Algorithm column in the csv report
		cv.runExperiment(mimlbagging);
		System.out.println(report.toString(cv));
		report.saveReport(report.toCSV(cv));

		System.out.println(
				"-Seventh example cross-validation using MIMLtoML transformation with medoidTransformation:\n");
		ConfigParameters.setAlgorithmName("toML_MDT_BRkNN"); // value for Algorithm column in the csv report
		cv.runExperiment(mimltoml_medoid);
		System.out.println(report.toString(cv));
		report.saveReport(report.toCSV(cv));

		System.out.println("The program has finished.");
	}
}
