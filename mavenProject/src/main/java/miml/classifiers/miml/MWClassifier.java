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

import com.mathworks.toolbox.javabuilder.MWCellArray;
import com.mathworks.toolbox.javabuilder.MWException;
import com.mathworks.toolbox.javabuilder.MWNumericArray;

import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import miml.data.MWTranslator;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;

/**
 * <p>
 * Class to execute Matlab MIML classifiers.
 * </p>
 * 
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20220106
 */
public abstract class MWClassifier extends MIMLClassifier {

	/** For serialization. */
	private static final long serialVersionUID = 949767245365319772L;

	/** Wrapper for Matlab data types. */
	protected MWTranslator wrapper;

	/** It will store the trained classifier. */
	protected Object[] classifier = null;

	@Override
	protected void buildInternal(MIMLInstances trainingSet) throws Exception {
		wrapper = new MWTranslator(trainingSet);

		MWCellArray train_bags = wrapper.getBags();
		MWNumericArray train_targets = wrapper.getLabels();

		classifier = trainMWClassifier(train_bags, train_targets);

	}

	@Override
	protected MultiLabelOutput makePredictionInternal(MIMLBag aBag) throws Exception, InvalidDataException {
		MWCellArray train_bags = wrapper.getBags();
		MWNumericArray train_targets = wrapper.getLabels();
		MWNumericArray test_bag = wrapper.getBagAsArray(aBag);

		Object[] prediction = predictMWClassifier(train_bags, train_targets, test_bag);

		// Only if outputs were being considered
		// double outputs[] = null;
		// if (prediction[0] != null) {
		// if (prediction[0] instanceof MWNumericArray) {
		// outputs = ((MWNumericArray) prediction[0]).getDoubleData();
		// }

		double pre_labels[] = null;
		if (prediction[1] instanceof MWNumericArray) {
			pre_labels = ((MWNumericArray) prediction[1]).getDoubleData();
		}

		boolean bipartition[] = new boolean[numLabels];
		double confidences[] = new double[numLabels];
		for (int l = 0; l < numLabels; l++) {
			bipartition[l] = (pre_labels[l] < 0) ? false : true;
			confidences[l] = bipartition[l] ? 1.0 : 0;
		}

		MultiLabelOutput finalDecision = null;

		finalDecision = new MultiLabelOutput(bipartition, confidences);
		return finalDecision;
	}

	/**
	 * Trains a Matlab classifier. Returns the classifier model in an array of
	 * Object.
	 * 
	 * @param train_bags    bags in the MIMLInstances dataset in the format of a
	 *                      nBagsx1 MWCellArray in which the ith bag is stored in
	 *                      aCellArray{i,1}. Each bag is a nInstxnAttributes array
	 *                      of double values.
	 * @param train_targets Label associations of all bags in the MIMLInstances
	 *                      dataset in the format of a nLabelsxnBags MWNumericArray
	 *                      of double. If the ith bag belongs to the jth label, then
	 *                      aDoubleArray(j,i) equals +1, otherwise train_target(j,i)
	 *                      equals -1.
	 * @throws MWException To be handled.
	 * @return An array of object. The number of elements will be the same as
	 *         elements returns function classifier.CLASSIFIER_run_train.
	 */
	protected abstract Object[] trainMWClassifier(MWCellArray train_bags, MWNumericArray train_targets)
			throws MWException;

	/**
	 * Performs a prediction on a test bag.
	 * 
	 * @param train_bags    Bags in the MIMLInstances dataset in the format of a
	 *                      nBagsx1 MWCellArray in which the ith bag is stored in
	 *                      aCellArray{i,1}. Each bag is a nInstxnAttributes array
	 *                      of double values.
	 * @param train_targets Label associations of all bags in the MIMLInstances
	 *                      dataset in the format of a nLabelsxnBags MWNumericArray
	 *                      of double. If the ith bag belongs to the jth label, then
	 *                      aDoubleArray(j,i) equals +1, otherwise train_target(j,i)
	 *                      equals -1.
	 * @param test_bag      A test bag. It will be a MIMLBag in the format of a
	 *                      nInstxnAttributes MWNumericArray of double.
	 * @return An array of 2 Object:
	 *         <ul>
	 *         <li>Object[0] is a nLabelsx1 array of double containing the
	 *         probability of the testing instance belonging to each label.</li>
	 *         <li>Object[1] is a nLabelsx1 array of double containing a bipartition
	 *         being 1 if the label is relevant or -1 otherwise.</li>
	 *         </ul>
	 * @throws MWException To be handled.
	 */
	protected abstract Object[] predictMWClassifier(MWCellArray train_bags, MWNumericArray train_targets,
			MWNumericArray test_bag) throws MWException;
}
