/*
 *    This program is free software; you can redistribute it and/or modify
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

import miml.data.MLSave;
import miml.data.partitioning.iterative.IterativeCrossValidation;
import miml.data.partitioning.iterative.IterativeTrainTest;
import miml.data.partitioning.powerset.LabelPowersetCrossValidation;
import miml.data.partitioning.powerset.LabelPowersetTrainTest;
import miml.data.partitioning.random.RandomCrossValidation;
import miml.data.partitioning.random.RandomTrainTest;
import mulan.data.MultiLabelInstances;
import weka.core.Utils;

/**
 * Class to split a multi-output dataset into partitions for cross-validation or
 * train-test. This class is able to work on multi-label, multi-instance
 * multi-label, and multi-view multi-label.
 * 
 * @author Eva Gibaja
 * @version 20201029
 */
public class GeneratePartitions {

	/**
	 * Shows the help on command line.
	 */
	public static void showUse() {
		System.out.println("Program parameters:");
		// Files
		System.out.println("\t -f file.arff");
		System.out.println("\t -x file.xml");

		// Action
		System.out.println("\t -t|c value");
		System.out.println("\t\t -t double_percentage ->  train-test and train percentage");
		System.out.println("\t\t -c integer_nFolds  -> cross-validation and number of folds");

		// Options
		System.out.println("\t -s 0|1|2");
		System.out.println("\t\t -s 0 ->  random  stratification. For classification and regression (by default)");
		System.out.println("\t\t -s 1 ->  iterative stratification. Just for classification");
		System.out.println("\t\t -s 2 ->  label powerset stratification. Just for classification");

		// Output
		System.out.println("\t -o OutputFile (without extension)");
		System.out.println("\t\t train-test -> OutputFile_train.arff and OutputFile_test.arff");
		System.out.println("\t\t cross-validation -> OutputFile_1.arff ... OutputFile_nFolds.arff");

		// Example
		System.out.println("Examples:");

		System.out.println("java -jar GeneratePartitions -f toy.arff -x toy.xml -t 80 -s 1  -o outputFolder"
				+ File.separator + "toy");
		System.out.println("java -jar GeneratePartitions -f data" + File.separator + "toy.arff -x data" + File.separator
				+ "toy.xml -c 10 -s 0 -o toy");

		System.exit(-1);
	}

	/**
	 * Main method.
	 * 
	 * @param args
	 *            Command line arguments.
	 *            <ul>
	 *            <li>-f filename.arff -&gt; name of the filename to be
	 *            partitioned</li>
	 *            <li>-x file.xml</li>
	 *            <li>-[t|c] value
	 *            <ul>
	 *            <li>-t double_percentage -&gt; train-test and tranin
	 *            percentage</li>
	 *            <li>-c integer_nFolds -&gt; cross-validation and number of
	 *            folds</li>
	 *            </ul>
	 *            </li>
	 *            <li>-s 0|1|2
	 *            <ul>
	 *            <li>-s 0 -&gt; random stratification (by default)</li>
	 *            <li>-s 1 -&gt; iterative stratification</li>
	 *            <li>-s 2 -&gt; label powerset stratification</li>
	 *            </ul>
	 *            *
	 *            <li>-o OutputFile (without extension)
	 *            <ul>
	 *            <li>train-test -&gt; OutputFile_train.arff and
	 *            OutputFile_test.arff</li>
	 *            <li>cross-validation -&gt; OutputFile_1.arff ...
	 *            OutputFile_nFolds.arff</li>
	 *            </ul>
	 *            </ul>
	 * @throws Exception
	 *             To be handled.
	 */
	public static void main(String[] args) throws Exception {
		MultiLabelInstances[] partitions = null;
		MultiLabelInstances[][] rounds = null;

		MultiLabelInstances mlDataSet = null;

		// Gets option values
		String arffName = Utils.getOption("f", args);
		String sTargets = Utils.getOption("x", args);
		String percentage = Utils.getOption("t", args);
		String folds = Utils.getOption("c", args);
		String outputFile = Utils.getOption("o", args);
		String stratification = Utils.getOption("s", args);

		// Error checking
		boolean error = false;
		if (arffName.equals("")) {
			System.out.println("Error: Empty .arff file");
			error = true;
		}
		if (sTargets.equals("")) {
			System.out.println("Error: Empty .xml file");
			error = true;
		}
		if (outputFile.equals("")) {
			System.out.println("Error: Empty output file");
			error = true;
		}
		if (percentage.equals("") && folds.equals("")) {
			System.out.println("Error: Unspecified -t or -c option");
			error = true;
		}

		if (error) {
			// Exits the program
			showUse();
			System.exit(-1);
		}

		// Loads the dataset
		mlDataSet = new MultiLabelInstances(arffName, sTargets);

		if (!percentage.equalsIgnoreCase("")) {
			// The percentage option is not empty
			// Train-test
			double percentageValue = Double.parseDouble(percentage);
			if (stratification.contentEquals("1")) {
				// Iterative stratification
				System.out.println("\nPerforming iterative train-test partitioning (" + percentage + " train)...");
				IterativeTrainTest engine = new IterativeTrainTest(mlDataSet);
				partitions = engine.split(percentageValue);
			} else if (stratification.contentEquals("2")) {
				// Labelpowerset stratification
				System.out.println("\nPerforming label powerset train-test partitioning (" + percentage + " train)...");
				LabelPowersetTrainTest engine = new LabelPowersetTrainTest(mlDataSet);
				partitions = engine.split(percentageValue);
			} else {
				// Random (default option)
				System.out.println("\nPerforming random train-test partitioning (" + percentage + " train)...");
				RandomTrainTest engine = new RandomTrainTest(mlDataSet);
				partitions = engine.split(percentageValue);

			}
			// Save the train-test partition
			String aux = new String(outputFile + "_" + percentage + "train");
			MLSave.saveArff(partitions[0], new String(aux + ".arff"));
			aux = new String(outputFile + "_" + (100 - (int) percentageValue) + "test");
			MLSave.saveArff(partitions[1], new String(aux + ".arff"));

		} else {
			// Cross-validation
			int foldsValue = Integer.parseInt(folds);
			if (stratification.contentEquals("1")) {
				// Iterative stratification
				System.out.println("\nPerforming iterative cross-validation partitioning (" + folds + " folds)...");
				IterativeCrossValidation engine = new IterativeCrossValidation(mlDataSet);
				partitions = engine.getFolds(foldsValue);
				rounds = engine.getRounds(foldsValue);
			} else if (stratification.contentEquals("2")) {
				// Labelpowerset stratification
				System.out
						.println("\nPerforming label powerset cross-validation partitioning (" + folds + " folds)...");
				LabelPowersetCrossValidation engine = new LabelPowersetCrossValidation(mlDataSet);
				partitions = engine.getFolds(foldsValue);
				rounds = engine.getRounds(foldsValue);

			} else {
				// Random (default option)
				System.out.println("\nPerforming random cross-validation partitioning (" + folds + " folds)...");
				RandomCrossValidation engine = new RandomCrossValidation(mlDataSet);
				partitions = engine.getFolds(foldsValue);
				rounds = engine.getRounds(foldsValue);
			}

			// Save partition (folds and rounds of train and test)
			for (int i = 0; i < partitions.length; i++) {

				// Save folds
				String aux = new String(outputFile + "_fold_" + (i + 1));
				MLSave.saveArff(partitions[i], aux + ".arff");

				// Save rounds
				aux = new String(outputFile + "_train_" + (i + 1));
				MLSave.saveArff(rounds[i][0], new String(aux + ".arff"));

				aux = new String(outputFile + "_test_" + (i + 1));
				MLSave.saveArff(rounds[i][1], new String(aux + ".arff"));

			}
		}

	}

}
