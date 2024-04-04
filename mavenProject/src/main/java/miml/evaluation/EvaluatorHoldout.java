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

package miml.evaluation;

import java.io.File;
import java.util.Date;
import java.util.concurrent.TimeUnit;

import org.apache.commons.configuration2.Configuration;

import miml.classifiers.miml.IMIMLClassifier;
import miml.core.ConfigParameters;
import miml.core.IConfiguration;
import miml.data.MIMLInstances;
import mulan.data.InvalidDataFormatException;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;

/**
 * Class that allow evaluate an algorithm applying a holdout method.
 *
 * @author Alvaro A. Belmonte
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20180630
 */
public class EvaluatorHoldout implements IConfiguration, IEvaluator<Evaluation> {

	/** The evaluation method used in holdout. */
	protected Evaluation evaluation;

	/** The data used in the experiment. */
	protected MIMLInstances trainData;

	/** The test data used in the experiment. */
	protected MIMLInstances testData;

	/** Train time in milliseconds. */
	protected long trainTime;

	/** Test time in milliseconds. */
	protected long testTime;

	/**
	 * Instantiates a new holdout evaluator with provided train and test partitions.
	 *
	 * @param trainData The train data used in the experiment.
	 * @param testData  The test data used in the experiment.
	 * @throws InvalidDataFormatException To be handled.
	 */
	public EvaluatorHoldout(MIMLInstances trainData, MIMLInstances testData) throws InvalidDataFormatException {
		this.trainData = trainData;
		this.testData = testData;
	}

	/**
	 * Instantiates a new holdout evaluator with random partitioning method.
	 *
	 * @param mimlDataSet     The dataset to be used.
	 * @param percentageTrain The percentage of train.
	 * @throws Exception If occur an error during holdout experiment.
	 */
	public EvaluatorHoldout(MIMLInstances mimlDataSet, double percentageTrain) throws Exception {
		this(mimlDataSet, percentageTrain, 1, 1);
	}

	/**
	 * Instantiates a new Holdout evaluator with a partitioning method and a seed.
	 *
	 * @param mimlDataSet     The dataset to be used.
	 * @param percentageTrain The percentage of train.
	 * @param seed            Seed for randomization.
	 * @param method          partitioning method.
	 *                        <ul>
	 *                        <li>1 random partitioning</li>
	 *                        <li>2 powerset partitioning</li>
	 *                        <li>3 iterative partitioning</li>
	 *                        </ul>
	 * @throws Exception If occur an error during holdout experiment.
	 */
	public EvaluatorHoldout(MIMLInstances mimlDataSet, double percentageTrain, int seed, int method) throws Exception {

		MIMLInstances[] partitions = MIMLInstances.splitData(mimlDataSet, percentageTrain, seed, method);
		this.trainData = partitions[0];
		this.testData = partitions[1];
	}

	/**
	 * No-argument constructor for xml configuration.
	 */
	public EvaluatorHoldout() {

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see evaluation.IEvaluator#runExperiment(mimlclassifier.MIMLClassifier)
	 */
	@Override
	public void runExperiment(IMIMLClassifier classifier) {

		Evaluator eval = new Evaluator();

		System.out.println("" + new Date() + ": " + "Building model");
		try {

			long startTime = System.nanoTime();
			classifier.build(trainData);
			long estimatedTime = System.nanoTime() - startTime;
			trainTime = TimeUnit.NANOSECONDS.toMillis(estimatedTime);

			System.out.println("" + new Date() + ": " + "Getting evaluation results");

			startTime = System.nanoTime();
			evaluation = eval.evaluate(classifier, testData, trainData);
			estimatedTime = System.nanoTime() - startTime;
			testTime = TimeUnit.NANOSECONDS.toMillis(estimatedTime);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Gets the time spent in training.
	 *
	 * @return The train time.
	 */
	public long getTrainTime() {
		return trainTime;
	}

	/**
	 * Gets the time spent in testing.
	 *
	 * @return The test time.
	 */
	public long getTestTime() {
		return testTime;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see evaluation.IEvaluator#getEvaluation()
	 */
	@Override
	public Evaluation getEvaluation() {
		return evaluation;
	}

	/**
	 * Gets the data used in the experiment.
	 *
	 * @return The data.
	 */
	public MIMLInstances getData() {
		return testData;
	}

	/*
	 * @see
	 * core.IConfiguration#configure(org.apache.commons.configuration.Configuration)
	 */
	@Override
	public void configure(Configuration configuration) {

		String arffFileTrain = configuration.subset("data").getString("trainFile");
		String xmlFileName = configuration.subset("data").getString("xmlFile");
		String arffFileTest = configuration.subset("data").getString("testFile");

		try {

			if (arffFileTest == null) {
				// if partitioning method is not provided it will be random partitioning
				String partitioning = configuration.getString("partitionMethod", "random");
				int method = 1; // by default random partitioning

				if (partitioning.equals("random"))
					method = 1;
				if (partitioning.equals("powerset"))
					method = 2;
				if (partitioning.equals("iterative"))
					method = 3;

				int seed = configuration.getInt("seed", 1);
				double percentageTrain = configuration.getDouble("percentageTrain", 80);
				MIMLInstances[] partitions = MIMLInstances.splitData(new MIMLInstances(arffFileTrain, xmlFileName),
						percentageTrain, seed, method);
				this.trainData = partitions[0];
				this.testData = partitions[1];
			} else {

				trainData = new MIMLInstances(arffFileTrain, xmlFileName);
				testData = new MIMLInstances(arffFileTest, xmlFileName);
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

		ConfigParameters.setDataFileName(new File(arffFileTrain).getName());

	}

}
