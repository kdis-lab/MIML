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
package miml.classifiers.miml;

import java.util.Date;

import miml.core.IConfiguration;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.classifier.MultiLabelOutput;
import mulan.core.ArgumentNullException;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializedObject;

/**
 * This java class is based on the mulan.data.Statistics.java class provided in
 * the Mulan java framework for multi-label learning <em>Tsoumakas, G., Katakis, I.,
 * Vlahavas, I. (2010) "Mining Multi-label Data", Data Mining and Knowledge
 * Discovery Handbook, O. Maimon, L. Rokach (Ed.), Springer, 2nd edition, 2010.</em>
 * Our contribution is mainly related with providing a framework to work with MIML data.
 * 
 * @author Ana I. Reyes
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @author Alvaro A. Belmonte
 * @version 20180619
 *
 *
 */
public abstract class MIMLClassifier implements IConfiguration, IMIMLClassifier {

	/** Generated Serial version UID. */
	private static final long serialVersionUID = -517275642740330327L;

	/** Boolean that indicate if the model has been initialized. */
	protected boolean isModelInitialized = false;
	/**
	 * The number of labels the learner can handle. The number of labels is
	 * determined from the training data when learner is build.
	 */
	protected int numLabels;
	/**
	 * An array containing the indexes of the label attributes within the
	 * {@link Instances} object of the training data in increasing order. The same
	 * order will be followed in the arrays of predictions given by each learner in
	 * the {@link MultiLabelOutput} object.
	 */
	protected int[] labelIndices;
	/**
	 * An array containing the names of the label attributes within the
	 * {@link Instances} object of the training data in increasing order. The same
	 * order will be followed in the arrays of predictions given by each learner in
	 * the {@link MultiLabelOutput} object.
	 */
	protected String[] labelNames;
	/**
	 * An array containing the indexes of the feature attributes within the
	 * {@link Instances} object of the training data in increasing order.
	 */
	protected int[] featureIndices;

	/** Whether debugging is on/off. */
	private boolean isDebug = false;

	/*
	 * (non-Javadoc)
	 * 
	 * @see mulan.classifier.MultiLabelLearner#isUpdatable()
	 */
	@Override
	public boolean isUpdatable() {
		/** as default learners are assumed not to be updatable. */
		return false;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see mulan.classifier.MultiLabelLearner#build(mulan.data.MultiLabelInstances)
	 */
	@Override
	public final void build(MultiLabelInstances trainingSet) throws Exception {

		if (trainingSet == null) {
			throw new ArgumentNullException("trainingSet");
		}

		isModelInitialized = false;

		numLabels = trainingSet.getNumLabels();
		labelIndices = trainingSet.getLabelIndices();
		labelNames = trainingSet.getLabelNames();
		featureIndices = trainingSet.getFeatureIndices();

		buildInternal(new MIMLInstances(trainingSet.getDataSet(), trainingSet.getLabelsMetaData()));
		isModelInitialized = true;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see mimlclassifier.IMIMLClassifier#build(data.MIMLInstances)
	 */
	@Override
	public final void build(MIMLInstances trainingSet) throws Exception {

		if (trainingSet == null) {
			throw new ArgumentNullException("trainingSet");
		}

		isModelInitialized = false;

		numLabels = trainingSet.getNumLabels();
		labelIndices = trainingSet.getLabelIndices();
		labelNames = trainingSet.getLabelNames();
		featureIndices = trainingSet.getFeatureIndices();

		buildInternal(new MIMLInstances(trainingSet.getDataSet(), trainingSet.getLabelsMetaData()));
		isModelInitialized = true;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see mimlclassifier.IMIMLClassifier#makePrediction(weka.core.Instance)
	 */
	@Override
	public final MultiLabelOutput makePrediction(Instance instance)
			throws Exception, InvalidDataException, ModelInitializationException {
		if (instance == null) {
			throw new ArgumentNullException("instance");
		}
		if (!isModelInitialized()) {
			throw new ModelInitializationException("The model has not been trained.");
		}
		
		instance.setValue(labelIndices[labelIndices.length-1], 1);
		
		return makePredictionInternal(new MIMLBag(instance));
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see mimlclassifier.IMIMLClassifier#setDebug(boolean)
	 */
	@Override
	public void setDebug(boolean debug) {
		isDebug = debug;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see mimlclassifier.IMIMLClassifier#makeCopy()
	 */
	@Override
	public IMIMLClassifier makeCopy() throws Exception {
		return (IMIMLClassifier) new SerializedObject(this).getObject();
	}

	/**
	 * Gets whether learner's model is initialized by
	 * {@link #build(MultiLabelInstances)}. This is used to check if
	 * {@link #makePrediction(weka.core.Instance)} can be processed.
	 * 
	 * @return <code>true</code> if the model has been initialized.
	 */
	protected boolean isModelInitialized() {
		return isModelInitialized;
	}

	/**
	 * Get whether debugging is turned on.
	 *
	 * @return <code>True</code> if debugging output is on
	 */
	public boolean getDebug() {
		return isDebug;
	}

	/**
	 * Writes the debug message string to the console output if debug for the
	 * learner is enabled.
	 *
	 * @param msg The debug message
	 */
	protected void debug(String msg) {
		if (!getDebug()) {
			return;
		}
		System.err.println("" + new Date() + ": " + msg);
	}

	/**
	 * Learner specific implementation of building the model from
	 * {@link MultiLabelInstances} training data set. This method is called from
	 * {@link #build(MultiLabelInstances)} method, where behavior common across all
	 * learners is applied.
	 *
	 * @param trainingSet The training data set.
	 * @throws Exception if learner model was not created successfully.
	 */
	protected abstract void buildInternal(MIMLInstances trainingSet) throws Exception;

	/**
	 * Learner specific implementation for predicting on specified data based on
	 * trained model. This method is called from
	 * {@link #makePrediction(weka.core.Instance)} which guards for model
	 * initialization and apply common handling/behavior.
	 *
	 * @param instance The data instance to predict on.
	 * @return The output of the learner for the given instance.
	 * @throws Exception            If an error occurs while making the prediction.
	 * @throws InvalidDataException If specified instance data is invalid and can
	 *                              not be processed by the learner.
	 */
	protected abstract MultiLabelOutput makePredictionInternal(MIMLBag instance) throws Exception, InvalidDataException;

}