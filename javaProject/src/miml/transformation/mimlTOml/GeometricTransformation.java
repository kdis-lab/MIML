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
package miml.transformation.mimlTOml;

import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import mulan.data.MultiLabelInstances;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 
 * Class that performs a geometric transformation to convert a MIMLInstances
 * class to MultiLabelInstances. Each Bag is transformed into a single Instance
 * being the value of each attribute the geometric centor of its max and min
 * values computed as (min_value+max_value)/2.
 * 
 * @author Ana I. Reyes Melero
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20180610
 *
 */
public class GeometricTransformation extends MIMLtoML {

	/** For serialization */
	private static final long serialVersionUID = 2536049747712271218L;

	/**
	 * Constructor
	 * 
	 * @param dataset MIMLInstances dataset.
	 * @throws Exception To be handled in an upper
	 *                   level.
	 */
	public GeometricTransformation(MIMLInstances dataset) throws Exception {
		this.dataset = dataset;
		this.prepareTemplate();
		template.setRelationName(dataset.getDataSet().relationName() + "_geometric_transformation");
	}

	public GeometricTransformation() throws Exception {
		super();
	}

	@Override
	public MultiLabelInstances transformDataset() throws Exception {

		Instances newData = new Instances(template);
		int labelIndices[] = dataset.getLabelIndices();
		Instance newInst = new DenseInstance(newData.numAttributes());
		newInst.setDataset(newData); // Sets the reference to the dataset

		// For all bags in the dataset
		double nBags = dataset.getNumBags();
		for (int i = 0; i < nBags; i++) {
			// retrieves a bag
			MIMLBag bag = dataset.getBag(i);
			// sets the bagLabel
			newInst.setValue(0, bag.value(0));

			// retrieves instances (relational value) for each bag
			Instances instances = bag.getBagAsInstances();
			// for all attributes in bag
			for (int j = 0, attIdx = 1; j < instances.numAttributes(); j++, attIdx++) {
				double[] minimax = minimax(instances, j);
				double value = (minimax[0] + minimax[1]) / 2.0;
				newInst.setValue(attIdx, value);
			}

			// inserts label information into the instance
			for (int j = 0; j < labelIndices.length; j++) {
				newInst.setValue(updatedLabelIndices[j], dataset.getBag(i).value(labelIndices[j]));
			}

			newData.add(newInst);
		}
		return new MultiLabelInstances(newData, dataset.getLabelsMetaData());
	}

	@Override
	public MultiLabelInstances transformDataset(MIMLInstances dataset) throws Exception {

		this.dataset = dataset;
		this.prepareTemplate();
		template.setRelationName(dataset.getDataSet().relationName() + "_geometric_transformation");

		Instances newData = new Instances(template);
		int labelIndices[] = dataset.getLabelIndices();
		Instance newInst = new DenseInstance(newData.numAttributes());
		newInst.setDataset(newData); // Sets the reference to the dataset

		// For all bags in the dataset
		double nBags = dataset.getNumBags();
		for (int i = 0; i < nBags; i++) {
			// retrieves a bag
			MIMLBag bag = dataset.getBag(i);
			// sets the bagLabel
			newInst.setValue(0, bag.value(0));

			// retrieves instances (relational value) for each bag
			Instances instances = bag.getBagAsInstances();
			// for all attributes in bag
			for (int j = 0, attIdx = 1; j < instances.numAttributes(); j++, attIdx++) {
				double[] minimax = minimax(instances, j);
				double value = (minimax[0] + minimax[1]) / 2.0;
				newInst.setValue(attIdx, value);
			}

			// inserts label information into the instance
			for (int j = 0; j < labelIndices.length; j++) {
				newInst.setValue(updatedLabelIndices[j], dataset.getBag(i).value(labelIndices[j]));
			}

			newData.add(newInst);
		}
		return new MultiLabelInstances(newData, dataset.getLabelsMetaData());
	}

	@Override
	public Instance transformInstance(MIMLBag bag) throws Exception {
		int labelIndices[] = dataset.getLabelIndices();
		Instance newInst = new DenseInstance(template.numAttributes());

		// sets the bagLabel
		newInst.setDataset(bag.dataset()); // Sets the reference to the dataset
		newInst.setValue(0, bag.value(0));

		// retrieves instances (relational value)
		Instances instances = bag.getBagAsInstances();
		// For all attributes in bag
		for (int j = 0, attIdx = 1; j < instances.numAttributes(); j++, attIdx++) {
			double[] minimax = minimax(instances, j);
			double value = (minimax[0] + minimax[1]) / 2.0;
			newInst.setValue(attIdx, value);
		}

		// Insert label information into the instance
		for (int j = 0; j < labelIndices.length; j++) {
			newInst.setValue(updatedLabelIndices[j], bag.value(labelIndices[j]));
		}

		return newInst;
	}

	public Instance transformInstance(MIMLInstances dataset, MIMLBag bag) throws Exception {

		this.dataset = dataset;
		this.prepareTemplate();
		template.setRelationName(dataset.getDataSet().relationName() + "_geometric_transformation");

		int labelIndices[] = dataset.getLabelIndices();
		Instance newInst = new DenseInstance(template.numAttributes());

		// sets the bagLabel
		newInst.setDataset(bag.dataset()); // Sets the reference to the dataset
		newInst.setValue(0, bag.value(0));

		// retrieves instances (relational value)
		Instances instances = bag.getBagAsInstances();
		// For all attributes in bag
		for (int j = 0, attIdx = 1; j < instances.numAttributes(); j++, attIdx++) {
			double[] minimax = minimax(instances, j);
			double value = (minimax[0] + minimax[1]) / 2.0;
			newInst.setValue(attIdx, value);
		}

		// Insert label information into the instance
		for (int j = 0; j < labelIndices.length; j++) {
			newInst.setValue(updatedLabelIndices[j], bag.value(labelIndices[j]));
		}

		return newInst;
	}
}
