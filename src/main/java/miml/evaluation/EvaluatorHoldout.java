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
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.apache.commons.configuration2.Configuration;

import miml.classifiers.miml.IMIMLClassifier;
import miml.core.ConfigParameters;
import miml.core.IConfiguration;
import miml.data.MIMLInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;

// TODO: Auto-generated Javadoc
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

	/** Seed for randomization */
	protected int seed = 1;

	/**
	 * Instantiates a new Holdout evaluator.
	 *
	 * @param trainData The train data used in the experiment.
	 * @param testData  The test data used in the experiment.
	 */
	public EvaluatorHoldout(MIMLInstances trainData, MIMLInstances testData) {
		this.trainData = trainData;
		this.testData = testData;
	}

	/**
	 * Instantiates a new Holdout evaluator.
	 *
	 * @param mimlDataSet     the dataset to be used
	 * @param percentageTrain the percentage of train
	 * @throws Exception if occur an error during holdout experiment
	 */
	public EvaluatorHoldout(MIMLInstances mimlDataSet, double percentageTrain) throws Exception {
		this.splitData(mimlDataSet, percentageTrain);
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
	 * Split data given a percentage.
	 *
	 * @param mimlDataSet The MIML dataset to be splited.
	 * @param percentageTrain The percentage (0-100) to be used in train.
	 * @throws Exception To be handled in an upper level.
	 */
	protected void splitData(MIMLInstances mimlDataSet, double percentageTrain) throws Exception {
		// splits the data set into train and test
		// copy of original data
		Instances dataSet = new Instances(mimlDataSet.getDataSet());
		dataSet.randomize(new Random(seed));

		// obtains train set
		RemovePercentage rmvp = new RemovePercentage();
		rmvp.setInvertSelection(true);
		rmvp.setPercentage(percentageTrain);
		rmvp.setInputFormat(dataSet);
		Instances trainDataSet = Filter.useFilter(dataSet, rmvp);

		// obtains test set
		rmvp = new RemovePercentage();
		rmvp.setPercentage(percentageTrain);
		rmvp.setInputFormat(dataSet);
		Instances testDataSet = Filter.useFilter(dataSet, rmvp);

		trainData = new MIMLInstances(trainDataSet, mimlDataSet.getLabelsMetaData());
		testData = new MIMLInstances(testDataSet, mimlDataSet.getLabelsMetaData());
	}
	
	/**
	 * Gets the seed used in the experiment.
	 *
	 * @return The seed.
	 */
	public int getSeed() {
		return seed;
	}

	/**
	 * Sets the seed used in the experiment.
	 *
	 * @param seed The new seed.
	 */
	public void setSeed(int seed) {
		this.seed = seed;
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
	 * (non-Javadoc)
	 * 
	 * @see
	 * core.IConfiguration#configure(org.apache.commons.configuration.Configuration)
	 */
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
				this.splitData(new MIMLInstances(arffFileTrain, xmlFileName),
						configuration.subset("data").getDouble("percentageTrain"));
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
