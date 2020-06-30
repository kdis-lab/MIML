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
import mulan.data.InvalidDataFormatException;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.MultipleEvaluation;
import weka.core.Instances;

/**
 * Class that allow evaluate an algorithm applying a cross-validation method.
 * 
 * @author Alvaro A. Belmonte
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20180630
 */
public class EvaluatorCV implements IConfiguration, IEvaluator<MultipleEvaluation> {

	/** The evaluation method used in cross-validation. */
	protected MultipleEvaluation multipleEvaluation;

	/** The data used in the experiment. */
	protected MIMLInstances data;

	/** The number of folds. */
	protected int numFolds;

	/** The seed for the partition. */
	protected int seed = 1;

	/** Train time in milliseconds. */
	protected long trainTime[];

	/** Test time in milliseconds. */
	protected long testTime[];

	/**
	 * Instantiates a new Holdout evaluator.
	 *
	 * @param data
	 *            The data used in the experiment.
	 * @param numFolds
	 *            The number of folds used in the cross-validation.
	 */
	public EvaluatorCV(MIMLInstances data, int numFolds) {
		this.data = data;
		this.numFolds = numFolds;
	}

	/**
	 * No-argument constructor for xml configuration.
	 */
	public EvaluatorCV() {

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see evaluation.IEvaluator#runExperiment(mimlclassifier.MIMLClassifier)
	 */
	@Override
	public void runExperiment(IMIMLClassifier classifier) throws Exception {

		System.out.println("" + new Date() + ": " + "Initializing cross validation");

		trainTime = new long[numFolds];
		testTime = new long[numFolds];
		Evaluation[] Evaluations = new Evaluation[numFolds];
		Instances workingSet = new Instances(data.getDataSet());
		workingSet.randomize(new Random(seed));

		for (int i = 0; i < numFolds; i++)
			try {

				System.out.println("Fold " + (i + 1) + "/" + numFolds);

				// gets train and test sets
				Instances train = workingSet.trainCV(numFolds, i);
				Instances test = workingSet.testCV(numFolds, i);
				MIMLInstances mlTrain = new MIMLInstances(train, data.getLabelsMetaData());
				MIMLInstances mlTest = new MIMLInstances(test, data.getLabelsMetaData());

				// train step and gets the train time
				IMIMLClassifier clone = classifier.makeCopy();
				clone.setDebug(true);
				long time_ini = System.nanoTime();
				clone.build(mlTrain);
				long time_fin = System.nanoTime();
				TimeUnit.NANOSECONDS.toMillis(time_fin - time_ini);
				trainTime[i] = TimeUnit.NANOSECONDS.toMillis(time_fin - time_ini);

				// test step and gets the test time
				Evaluator eval = new Evaluator();
				time_ini = System.nanoTime();
				Evaluations[i] = eval.evaluate(clone, mlTest, mlTrain);

				time_fin = System.nanoTime();
				TimeUnit.NANOSECONDS.toMillis(time_fin - time_ini);
				testTime[i] = TimeUnit.NANOSECONDS.toMillis(time_fin - time_ini);

			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		multipleEvaluation = new MultipleEvaluation(Evaluations, data);
		multipleEvaluation.calculateStatistics();
	}
	
	/**
	 * Calculate the mean of given array.
	 *
	 * @param array The array with long values.
	 * @return The mean of all array's values.
	 */
	protected double meanArray(long array[])
	{
		double sum = 0;
		for (int i = 0; i < array.length; i++)
			sum += array[i];
		return sum / array.length;
		
	}
	
	/**
	 * Calculate the standard deviation of given array.
	 *
	 * @param array the array with long values.
	 * @return The standard deviation of all array's values.
	 */
	protected double stdArray(long array[])
	{
		double mean = meanArray(array);	    
		double sum = 0;
		for (int i=0; i< array.length; i++)
			sum += Math.pow((array[i] - mean), 2);
		return Math.sqrt(sum / array.length);
	}
	

	/**
	 * Gets the time spent in training in each fold.
	 *
	 * @return The train time.
	 */
	public long[] getTrainTime() {
		return trainTime;
	}

	/**
	 * Gets the time spent in testing in each fold.
	 *
	 * @return The test time.
	 */
	public long[] getTestTime() {
		return testTime;
	}

	/**
	 * Gets the number of folds used in the experiment.
	 *
	 * @return The number of folds.
	 */
	public int getNumFolds() {
		return numFolds;
	}

	/**
	 * Sets the number of folds used in the experiment.
	 *
	 * @param numFolds The new number of folds.
	 */
	public void setNumFolds(int numFolds) {
		this.numFolds = numFolds;
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
	 * @param seed The new seed
	 */
	public void setSeed(int seed) {
		this.seed = seed;
	}

	/**
	 * Gets the average time of all folds in train.
	 *
	 * @return The average time of all folds.
	 */
	public double getAvgTrainTime() {
		return meanArray(trainTime);
	}

	/**
	 * Gets the average time of all folds in test.
	 *
	 * @return The average time of all folds.
	 */
	public double getAvgTestTime() {
		return meanArray(testTime);
	}

    /**
     * Gets the standard deviation time of all folds in train.
     *
     * @return The standard deviation time of all folds.
     */
    public double getStdTrainTime()
    {   	
       return stdArray(trainTime);	
    }

    /**
     * Gets the standard deviation time of all folds in test.
     *
     * @return The standard deviation time of all folds.
     */
    public double getStdTestTime()
    {   	
       return stdArray(testTime);	
    }

	/*
	 * (non-Javadoc)
	 * 
	 * @see evaluation.IEvaluator#getEvaluation()
	 */
	@Override
	public MultipleEvaluation getEvaluation() {
		return multipleEvaluation;
	}

	/**
	 * Gets the data used in the experiment.
	 *
	 * @return The data.
	 */
	public MIMLInstances getData() {
		return data;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * core.IConfiguration#configure(org.apache.commons.configuration.Configuration)
	 */
	@Override
	public void configure(Configuration configuration) {

		numFolds = configuration.getInt("numFolds", 5);
		seed = configuration.getInt("seed", 1);

		String arffFile = configuration.subset("data").getString("file");
		String xmlFileName = configuration.subset("data").getString("xmlFile");

		try {
			data = new MIMLInstances(arffFile, xmlFileName);
		} catch (InvalidDataFormatException e) {
			e.printStackTrace();
		}
		

		ConfigParameters.setDataFileName(new File(arffFile).getName());

	}

}
