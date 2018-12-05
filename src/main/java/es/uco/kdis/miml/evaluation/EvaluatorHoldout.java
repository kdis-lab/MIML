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

package es.uco.kdis.miml.evaluation;

import java.util.Date;

import org.apache.commons.configuration2.Configuration;

import es.uco.kdis.miml.core.IConfiguration;
import es.uco.kdis.miml.data.MIMLInstances;
import es.uco.kdis.miml.mimlclassifier.IMIMLClassifier;
import mulan.data.InvalidDataFormatException;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;

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

	/** The evaluation. */
	private Evaluation evaluation;

	/** The train data. */
	private MIMLInstances trainData;

	/** The test data. */
	private MIMLInstances testData;

	/**
	 * Instantiates a new Holdout evaluator.
	 *
	 * @param trainData the train data used in the experiment
	 * @param testData  the test data used in the experiment
	 */
	public EvaluatorHoldout(MIMLInstances trainData, MIMLInstances testData) {
		this.trainData = trainData;
		this.testData = testData;
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
			classifier.build(trainData);
			System.out.println("" + new Date() + ": " + "Getting evaluation results");
			evaluation = eval.evaluate(classifier, testData, trainData);
		} catch (Exception e) {
			e.printStackTrace();
		}
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
	 * Gets the data used for evaluate the measures.
	 *
	 * @return the data
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
		String arffFileTest = configuration.subset("data").getString("testFile");
		String xmlFileName = configuration.subset("data").getString("xmlFile");

		try {
			trainData = new MIMLInstances(arffFileTrain, xmlFileName);
			testData = new MIMLInstances(arffFileTest, xmlFileName);
		} catch (InvalidDataFormatException e) {
			e.printStackTrace();
		}

	}

}
