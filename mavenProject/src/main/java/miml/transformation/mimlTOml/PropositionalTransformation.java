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

import java.io.Serializable;

import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * 
 * Class that performs a propositionalTransformation to convert a MIMLInstances
 * dataset to MultiLabelInstances. This transformation transforms each Bag into
 * a set if instances, one for each instance in the bag of the instances in the
 * bag.
 * 
 * @author Eva Gibaja
 * @version 20210614
 *
 */
public class PropositionalTransformation implements Serializable {

	/** For serialization. */
	private static final long serialVersionUID = 1L;

	/** Array of updated label indices. */
	protected int updatedLabelIndices[];

	/** Template to store Instances. */
	protected Instances template = null;

	/** Original data set of MIMLInstances. */
	protected MIMLInstances dataset = null;

	/** Filter */
	protected Remove removeFilter;

	/** Whether bag attribute will be included in the transformed data */
	protected boolean includeBagId = false;

	/**
	 * Constructor.
	 * 
	 * @param dataset MIMLInstances dataset.
	 * @throws Exception To be handled in an upper level.
	 */
	public PropositionalTransformation(MIMLInstances dataset) throws Exception {
		this.dataset = dataset;
		this.prepareTemplate();
		template.setRelationName(dataset.getDataSet().relationName() + "_propositional_transformation");
	}

	/**
	 * Returns the value of includeBagId property.
	 * 
	 * @return The value of includeBagId property.
	 */

	public boolean isIncludeBagId() {
		return includeBagId;
	}

	/**
	 * Sets the value for includeBagId property.
	 * 
	 * @param includeBagId if true the bagId will be included in the transformed
	 *                     data.
	 */
	public void setIncludeBagId(boolean includeBagId) {
		this.includeBagId = includeBagId;
	}

	/**
	 * Constructor.
	 * 
	 * @param dataset      MIMLInstances dataset.
	 * @param includeBagId true if the bagId will be included in the transformed
	 *                     dataset
	 * 
	 * @throws Exception To be handled in an upper level.
	 */
	public PropositionalTransformation(MIMLInstances dataset, boolean includeBagId) throws Exception {
		this(dataset);
		this.includeBagId = includeBagId;
	}

	public MultiLabelInstances transformDataset() throws Exception {
		Instances newData = new Instances(template);
		int labelIndices[] = dataset.getLabelIndices();

		// For all bags in the dataset
		double nBags = dataset.getNumBags();
		for (int b = 0; b < nBags; b++) {
			// retrieves a bag
			MIMLBag bag = dataset.getBag(b);
			Instances bagAsInstances = bag.getBagAsInstances();
			Instance newInst = null;

			// For all instances in the bag
			for (int i = 0; i < bag.getNumInstances(); i++) {
				newInst = new DenseInstance(newData.numAttributes());

				// Sets the reference to the dataset
				newInst.setDataset(newData);

				// sets the bagLabel
				newInst.setValue(0, bag.value(0));

				// gets relational information of instance i
				Instance relational = bagAsInstances.instance(i);

				// for all attributes in the bag, attIdx = 1 to consider bagId
				for (int j = 0, attIdx = 1; j < relational.numAttributes(); j++, attIdx++) {
					newInst.setValue(attIdx, relational.value(j));
				}

				// inserts label information into the instance
				for (int j = 0; j < labelIndices.length; j++) {
					newInst.setValue(updatedLabelIndices[j], dataset.getBag(b).value(labelIndices[j]));
				}
				newData.add(newInst);
			}
		}
		if (includeBagId)
			return new MultiLabelInstances(newData, dataset.getLabelsMetaData());
		else
			return removeBagId(new MultiLabelInstances(newData, dataset.getLabelsMetaData()));
	}

	public MultiLabelInstances transformDataset(MIMLInstances dataset) throws Exception {

		this.dataset = dataset;
		this.prepareTemplate();
		template.setRelationName(dataset.getDataSet().relationName() + "_propositional_transformation");

		return transformDataset();
	}

	public MultiLabelInstances transformInstance(MIMLBag bag) throws Exception {

		int labelIndices[] = dataset.getLabelIndices();
		Instances result = new Instances(template, 0);

		Instances bagAsInstances = bag.getBagAsInstances();

		for (int i = 0; i < bagAsInstances.numInstances(); i++) {
			Instance newInst = new DenseInstance(template.numAttributes());
			// Sets the reference to the dataset
			newInst.setDataset(bag.dataset());

			// sets the bagLabel
			newInst.setValue(0, bag.value(0));

			// For all attributes in bag
			for (int j = 0, attIdx = 1; j < bagAsInstances.numAttributes(); j++, attIdx++) {

				newInst.setValue(attIdx, bagAsInstances.instance(i).value(j));
			}

			// Insert label information into the instance
			for (int j = 0; j < labelIndices.length; j++) {
				newInst.setValue(updatedLabelIndices[j], bag.value(labelIndices[j]));
			}
			result.add(newInst);
		}

		if (includeBagId)
			return new MultiLabelInstances(result, dataset.getLabelsMetaData());
		else
			return removeBagId(new MultiLabelInstances(result, dataset.getLabelsMetaData()));

	}

	public MultiLabelInstances transformInstance(MIMLInstances dataset, MIMLBag bag) throws Exception {

		this.dataset = dataset;
		this.prepareTemplate();
		template.setRelationName(dataset.getDataSet().relationName() + "_propositional_transformation");

		return transformInstance(bag);
	}

	/**
	 * Prepares a template to perform the transformation from MIMLInstances to
	 * MultiLabelInstances. This template includes: the bag label attribute, all
	 * attributes in the relational attribute as independent attributes and label
	 * attributes. For instance, in the relation above, the resulting template is
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
	 * &#064;relation template<br>
	 * &#064;attribute id {bag1,bag2}<br>
	 * &#064;attribute f1 numeric<br>
	 * &#064;attribute f2 numeric<br>
	 * &#064;attribute f3 numeric<br>
	 * * &#064;attribute label1 {0,1}<br>
	 * &#064;attribute label2 {0,1}<br>
	 * &#064;attribute label3 {0,1}<br>
	 * &#064;attribute label4 {0,1}<br>
	 * 
	 * @throws Exception To be handled in an upper level.
	 */
	protected void prepareTemplate() throws Exception {
		int labelIndices[] = dataset.getLabelIndices();
		Instances bags = dataset.getDataSet();

		template = bags.attribute(1).relation().stringFreeStructure();
		// insert a bag label attribute at the begining
		Attribute bagLabel = bags.attribute(0);
		template.insertAttributeAt(bagLabel, 0);

		// Insert labels as attributes in the dataset
		updatedLabelIndices = new int[labelIndices.length];
		for (int i = 0; i < labelIndices.length; i++) {
			Attribute attr = bags.attribute(labelIndices[i]);
			updatedLabelIndices[i] = template.numAttributes();
			template.insertAttributeAt(attr, updatedLabelIndices[i]);
		}
	}

	/**
	 * Removes the bagId attribute in MultiLabelInstances.
	 * 
	 * @param mlDataSetWithBagId A MultiLabelInstances dataset corresponding with
	 *                           the propositional representation of MIML data being
	 *                           the first attribute the bagID.
	 * @return MultiLabelInstances without first bagIdAttribute
	 * @throws Exception To be handled in an upper level.
	 */
	public static MultiLabelInstances removeBagId(MultiLabelInstances mlDataSetWithBagId) throws Exception {
		Remove removeFilter;

		// Deletes bagIdAttribute from dataset
		removeFilter = new Remove();
		int indexToRemove[] = { 0 };
		removeFilter.setAttributeIndicesArray(indexToRemove);
		removeFilter.setInputFormat(mlDataSetWithBagId.getDataSet());
		Instances newData = Filter.useFilter(mlDataSetWithBagId.getDataSet(), removeFilter);
		MultiLabelInstances withoutBagId = new MultiLabelInstances(newData, mlDataSetWithBagId.getLabelsMetaData());

		return withoutBagId;
	}
}
