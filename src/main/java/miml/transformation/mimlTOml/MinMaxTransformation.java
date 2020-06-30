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

import java.util.ArrayList;

import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 
 * Class that performs a miniMaxc transformation to convert a MIMLInstances class
 * to MultiLabelInstances. Each Bag is transformed into a single Instance in
 * which, for each attribute of the bag, its min and max value are included. For
 * instance, For instance, in the relation above, the resulting template is
 * showed.
 * 
 * &#064;relation toy<br>
 * &#064;attribute id {bag1,bag2}<br>
 * &#064;attribute bag relational<br>
 * &#064;attribute f1 numeric<br>
 * &#064;attribute f2 numeric<br>
 * &#064;attribute f3 numeric<br>
 * &#064;end bag<br>
 * &#064;attribute label1 {0,1}<br>
 * &#064;attribute label2 {0,1}<br>
 * &#064;attribute label3 {0,1}<br>
 * &#064;attribute label4 {0,1}<br>
 * 
 * &#064;relation minMaxTransformation<br>
 * &#064;attribute id {bag1,bag2}<br>
 * &#064;attribute f1_min numeric<br>
 * &#064;attribute f1_max numeric<br>
 * &#064;attribute f2_min numeric<br>
 * &#064;attribute f2_max numeric<br>
 * &#064;attribute f3_min numeric<br>
 * &#064;attribute f3_max numeric<br>
 * * &#064;attribute label1 {0,1}<br>
 * &#064;attribute label2 {0,1}<br>
 * &#064;attribute label3 {0,1}<br>
 * &#064;attribute label4 {0,1}<br>
 * 
 * @author Ana I. Reyes Melero
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20180610
 *
 */
public class MinMaxTransformation extends MIMLtoML {

	/** For serialization */
	private static final long serialVersionUID = 4161911837824822046L;

	/**
	 * Constructor.
	 * 
	 * @param dataset MIMLInstances dataset.
	 * @throws Exception To be handled in an upper level.
	 */
	public MinMaxTransformation(MIMLInstances dataset) throws Exception {
		this.dataset = dataset;
		prepareTemplate();
		template.setRelationName(dataset.getDataSet().relationName() + "_minimax_transformation");
	}

	public MinMaxTransformation() throws Exception {
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
			// For all attributes in bag
			for (int j = 0, attIdx = 1; j < instances.numAttributes(); j++, attIdx++) {
				double[] minimax = minimax(instances, j);
				newInst.setValue(attIdx, minimax[0]);// minima value
				newInst.setValue(attIdx + instances.numAttributes(), minimax[1]);// maxima
																					// value);
			}
			// Copy label information into the dataset
			for (int j = 0; j < labelIndices.length; j++) {
				newInst.setValue(updatedLabelIndices[j], bag.value(labelIndices[j]));
			}
			newData.add(newInst);

		}
		return new MultiLabelInstances(newData, dataset.getLabelsMetaData());
	}

	@Override
	public MultiLabelInstances transformDataset(MIMLInstances dataset) throws Exception {

		this.dataset = dataset;
		prepareTemplate();
		template.setRelationName(dataset.getDataSet().relationName() + "_minimax_transformation");

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
			// For all attributes in bag
			for (int j = 0, attIdx = 1; j < instances.numAttributes(); j++, attIdx++) {
				double[] minimax = minimax(instances, j);
				newInst.setValue(attIdx, minimax[0]);// minima value
				newInst.setValue(attIdx + instances.numAttributes(), minimax[1]);// maxima
																					// value);
			}
			// Copy label information into the dataset
			for (int j = 0; j < labelIndices.length; j++) {
				newInst.setValue(updatedLabelIndices[j], bag.value(labelIndices[j]));
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
			newInst.setValue(attIdx, minimax[0]);// minima value
			newInst.setValue(attIdx + instances.numAttributes(), minimax[1]);// maxima
																				// value);
		}

		// Insert label information into the instance
		for (int j = 0; j < labelIndices.length; j++) {
			newInst.setValue(updatedLabelIndices[j], bag.value(labelIndices[j]));
		}

		return newInst;
	}

	public Instance transformInstance(MIMLInstances dataset, MIMLBag bag) throws Exception {
		this.dataset = dataset;
		prepareTemplate();
		template.setRelationName(dataset.getDataSet().relationName() + "_minimax_transformation");

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
			newInst.setValue(attIdx, minimax[0]);// minima value
			newInst.setValue(attIdx + instances.numAttributes(), minimax[1]);// maxima
																				// value);
		}

		// Insert label information into the instance
		for (int j = 0; j < labelIndices.length; j++) {
			newInst.setValue(updatedLabelIndices[j], bag.value(labelIndices[j]));
		}

		return newInst;
	}

	@Override
	protected void prepareTemplate() throws Exception {
		int attrIndex = 0;

		ArrayList<Attribute> attributes = new ArrayList<Attribute>();
		// insert a bag label attribute at the begining
		Attribute attr = dataset.getDataSet().attribute(0);
		attributes.add(attr);

		// Adds attributes for min
		Instances bags = dataset.getDataSet().attribute(1).relation().stringFreeStructure();
		;
		for (int i = 0; i < dataset.getNumAttributesInABag(); i++) {
			attr = new Attribute("min_" + bags.attribute(i).name());
			attributes.add(attr);
			attrIndex++;
		}

		// Adds attributes for max
		for (int i = 0; i < dataset.getNumAttributesInABag(); i++) {
			attr = new Attribute("max_" + bags.attribute(i).name());
			attributes.add(attr);
			attrIndex++;
		}

		// Insert labels as attributes in the dataset
		int labelIndices[] = dataset.getLabelIndices();
		updatedLabelIndices = new int[labelIndices.length];
		ArrayList<String> values = new ArrayList<String>(2);
		values.add("0");
		values.add("1");
		for (int i = 0; i < labelIndices.length; i++) {
			attr = new Attribute(dataset.getDataSet().attribute(labelIndices[i]).name(), values);
			attributes.add(attr);
			attrIndex++;
			updatedLabelIndices[i] = attrIndex;
		}

		template = new Instances("template2n", attributes, 0);

	}

}
