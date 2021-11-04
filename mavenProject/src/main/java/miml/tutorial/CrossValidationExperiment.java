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

import miml.classifiers.miml.lazy.BRkNN_MIMLWrapper;
import miml.classifiers.miml.lazy.DistanceFunction_MIMLWrapper;
import miml.classifiers.miml.lazy.MIMLkNN;
import miml.classifiers.miml.mimlTOmi.MIMLClassifierToMI;
import miml.classifiers.miml.mimlTOmi.MIMLLabelPowerset;
import miml.classifiers.miml.mimlTOml.MIMLClassifierToML;
import miml.classifiers.miml.neural.MIMLRBF_MIMLWrapper;
import miml.core.distance.AverageHausdorff;
import miml.core.distance.MaximalHausdorff;
import miml.data.MIMLInstances;
import miml.evaluation.EvaluatorCV;
import miml.report.BaseMIMLReport;
import miml.transformation.mimlTOml.GeometricTransformation;
import mulan.classifier.lazy.DMLkNN;
import weka.classifiers.mi.SimpleMI;
import weka.core.Utils;

/**
 * Class implementing an example of using cross-validation with different
 * kinds of classifier.
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
				+ File.separator + "miml_birds.xml -r output"+ File.separator + "miml_birds_report.csv");
		System.exit(-1);
	}
	
	public static void main(String[] args) throws Exception {

		//-f  data/birds.arff -x data/birds.xml -r results/report.csv
		
		String arffFileName = Utils.getOption("f", args);
		String xmlFileName = Utils.getOption("x", args);
		String reportFileName = Utils.getOption("r", args);

		// Loads the dataset
		System.out.println("Loading the dataset...");
		MIMLInstances mimlDataSet = new MIMLInstances(arffFileName, xmlFileName);

		// MIML report
		BaseMIMLReport report = new BaseMIMLReport(null, reportFileName, true, true, false);

		// Cross-validation evaluator
		EvaluatorCV cv = new EvaluatorCV(mimlDataSet, 5);

		// Load classifiers
		System.out.println("Loading classifiers...");
		MIMLkNN mimlknn = new MIMLkNN(new MaximalHausdorff(mimlDataSet));
		MIMLClassifierToMI mimltomi = new MIMLClassifierToMI(new MIMLLabelPowerset(new SimpleMI()));
		MIMLClassifierToML mimltoml = new MIMLClassifierToML(new DMLkNN(), new GeometricTransformation());				
		BRkNN_MIMLWrapper brknnwrapper = new BRkNN_MIMLWrapper(new DistanceFunction_MIMLWrapper(new AverageHausdorff()));			
		MIMLRBF_MIMLWrapper mimlrbf = new MIMLRBF_MIMLWrapper(0.1, 0.6);
		
		System.out.println("\n");

		System.out.println("-First example cross-validation using MIMLkNN:\n");
		cv.runExperiment(mimlknn);
		System.out.println(report.toString(cv) + "\n\n");
		report.saveReport(report.toCSV(cv));

		System.out.println("-Second example cross-validation using MIMLtoMI transformation:\n");
		cv.runExperiment(mimltomi);
		System.out.println(report.toString(cv) + "\n\n");
		report.saveReport(report.toCSV(cv));

		System.out.println("-Third example cross-validation using MIMLtoML transformation:\n");
		cv.runExperiment(mimltoml);
		System.out.println(report.toString(cv));
		report.saveReport(report.toCSV(cv));
		
		System.out.println("-Fourth example cross-validation using BRkNN_MIMLWrapper:\n");
		cv.runExperiment(brknnwrapper);
		System.out.println(report.toString(cv));
		report.saveReport(report.toCSV(cv));

		System.out.println("-Fifth example cross-validation using MIMLRBF_MIMLWrapper:\n");
		cv.runExperiment(mimlrbf);
		System.out.println(report.toString(cv));
		report.saveReport(report.toCSV(cv));	
		
		System.out.println("The program has finished.");
	}
}
