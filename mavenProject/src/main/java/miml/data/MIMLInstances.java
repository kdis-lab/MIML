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
package miml.data;

import java.util.ArrayList;
import miml.data.partitioning.CrossValidationBase;
import miml.data.partitioning.iterative.IterativeCrossValidation;
import miml.data.partitioning.iterative.IterativeTrainTest;
import miml.data.partitioning.powerset.LabelPowersetCrossValidation;
import miml.data.partitioning.powerset.LabelPowersetTrainTest;
import miml.data.partitioning.random.RandomCrossValidation;
import miml.data.partitioning.random.RandomTrainTest;
import mulan.data.InvalidDataFormatException;
import mulan.data.LabelsMetaData;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 
 * Class inheriting from MultiLabelnstances to represent MIML data.
 * 
 * @author Ana I. Reyes Melero
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20190315
 *
 */
public class MIMLInstances extends MultiLabelInstances {

	/** Generated Serial version UID. */
	private static final long serialVersionUID = 1L;

	/**
	 * Constructor.
	 * 
	 * @param dataSet              A dataset of {@link Instances} with relational
	 *                             information.
	 * @param xmlLabelsDefFilePath Path of .xml file with information about labels.
	 * @throws InvalidDataFormatException To be handled in an upper level.
	 */
	public MIMLInstances(Instances dataSet, String xmlLabelsDefFilePath) throws InvalidDataFormatException {
		super(dataSet, xmlLabelsDefFilePath);
	}

	/**
	 * Constructor.
	 * 
	 * @param dataSet        A dataset of {@link Instances} with relational
	 *                       information.
	 * @param labelsMetaData Information about labels.
	 * @throws InvalidDataFormatException To be handled in an upper level.
	 */
	public MIMLInstances(Instances dataSet, LabelsMetaData labelsMetaData) throws InvalidDataFormatException {
		super(dataSet, labelsMetaData);
	}

	/**
	 * Constructor.
	 * 
	 * @param arffFilePath         Path of .arff file with Instances.
	 * @param xmlLabelsDefFilePath Path of .xml file with information about labels.
	 * @throws InvalidDataFormatException To be handled in an upper level.
	 */
	public MIMLInstances(String arffFilePath, String xmlLabelsDefFilePath) throws InvalidDataFormatException {
		super(arffFilePath, xmlLabelsDefFilePath);
	}

	/**
	 * Constructor.
	 * 
	 * @param arffFilePath       Path of .arff file with Instances.
	 * @param numLabelAttributes Number of label attributes.
	 * @throws InvalidDataFormatException To be handled in an upper level.
	 */
	public MIMLInstances(String arffFilePath, int numLabelAttributes) throws InvalidDataFormatException {
		super(arffFilePath, numLabelAttributes);
	}

	/**
	 * Constructor.
	 * 
	 * @param mimlDataSet A datasetof {@link MIMLInstances}.
	 * @throws InvalidDataFormatException To be handled in an upper level.
	 */
	public MIMLInstances(MIMLInstances mimlDataSet) throws InvalidDataFormatException {
		super(mimlDataSet.getDataSet(), mimlDataSet.getLabelsMetaData());
	}

	/**
	 * Constructor.
	 * 
	 * @param mlDataSet A multi-label datasetof .
	 * @throws InvalidDataFormatException To be handled in an upper level.
	 */
	public MIMLInstances(MultiLabelInstances mlDataSet) throws InvalidDataFormatException
	{
		this((MIMLInstances)mlDataSet);
	}
	
	/**
	 * Gets a {@link MIMLBag} (i.e. pattern) with a certain bagIndex.
	 * 
	 * @param bagIndex Index of the bag.
	 * @return Bag If bagIndex exceeds the number of bags in the dataset. To be
	 *         handled in an upper level.
	 * @throws Exception To be handled in an upper level.
	 */
	public MIMLBag getBag(int bagIndex) throws Exception {
		if (bagIndex > this.getNumBags())
			throw new Exception("Out of bounds bagIndex: " + bagIndex + ". Actual numberOfBags: " + this.getNumBags());
		else {
			Instances aux = this.getDataSet();
			DenseInstance aux1 = (DenseInstance) aux.get(bagIndex);
			return new MIMLBag(aux1);
		}
	}

	/**
	 * Gets a {@link MIMLBag} with a certain bagIndex in the form of a set of
	 * {@link Instances} considering just the relational information. Neither
	 * identification attribute of the Bag nor label attributes are included.
	 * 
	 * @param bagIndex Index of the bag.
	 * @return A bag or an instance from the index of the dataset.
	 * @throws Exception If bagIndex exceeds the number of bags in the dataset. To
	 *                   be handled in an upper level.
	 */
	public Instances getBagAsInstances(int bagIndex) throws Exception {
		if (bagIndex > this.getNumBags())
			throw new Exception("Out of bounds bagIndex: " + bagIndex + ". Actual numberOfBags: " + this.getNumBags());
		else {
			Instances bags = getBag(bagIndex).getBagAsInstances();
			return bags;
		}
	}

	/**
	 * Adds a Bag of Instances to the dataset.
	 *
	 * @param bag A Bag of Instances.
	 */
	public void addBag(MIMLBag bag) {
		this.getDataSet().add(bag);
	}

	/**
	 * Adds a Bag of Instances to the dataset in a certain index.
	 * 
	 * @param bag   A Bag of Instances.
	 * @param index The index to insert the Bag.
	 */
	public void addInstance(MIMLBag bag, int index) {
		this.getDataSet().add(index, bag);
	}

	/**
	 * Gets an instance of a bag.
	 * 
	 * @param bagIndex      The index of the bag in the data set.
	 * @param instanceIndex Is the index of the instance in the bag.
	 * @return Instance.
	 * @throws IndexOutOfBoundsException To be handled in an upper level.
	 */
	public Instance getInstance(int bagIndex, int instanceIndex) throws IndexOutOfBoundsException {
		return this.getDataSet().instance(bagIndex).relationalValue(1).instance(instanceIndex);
	}

	/**
	 * Gets the number of bags of the dataset.
	 * 
	 * @return The number of bags of the dataset.
	 */
	public int getNumBags() {
		return this.getNumInstances();
	}

	/**
	 * Gets the number of instances of a bag.
	 * 
	 * @param bagIndex A bag index.
	 * @return The number of instances of a bag
	 * @throws Exception To be handled in an upper level.
	 */
	public int getNumInstances(int bagIndex) throws Exception {
		return this.getBag(bagIndex).relationalValue(1).numInstances();
	}

	/**
	 * Gets the number of attributes of the dataset considering label attributes and
	 * the relational attribute with bags as a single attribute. For instance, in
	 * relation above, the returned value is 6.
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
	 * @return The number of attributes of the dataset.
	 */
	public int getNumAttributes() {
		return this.getDataSet().numAttributes();
	}

	/**
	 * Gets the total number of attributes of the dataset. This number includes
	 * attributes corresponding to labels. Instead the relational attribute, the
	 * number of attributes contained in the relational attribute is considered. For
	 * instance, in the relation above, the output of the method is 8.<br>
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
	 * 
	 * @return The total number of attributes of the dataset.
	 */
	public int getNumAttributesWithRelational() {
		return this.getDataSet().numAttributes() + this.getNumAttributesInABag() - 1;
	}

	/**
	 * Gets the number of attributes per bag. In MIML all bags have the same number
	 * of attributes.* For instance, in the relation above, the output of the method
	 * is 3.<br>
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
	 * @return The number of attributes per bag.
	 */
	public int getNumAttributesInABag() {
		return this.getDataSet().instance(0).relationalValue(1).numAttributes();
	}

	/**
	 * Returns the dataset as MultiLabelInstances.
	 * 
	 * @return MultiLabelInstances.
	 */
	public MultiLabelInstances getMLDataSet() {
		return this;
	}

	/**
	 * Adds an attribute to the relational attribute with value '?' at the last
	 * position.
	 * 
	 * @param newAttr The attribute to be added.
	 * @throws InvalidDataFormatException if occurred an error creating new dataset.
	 * 
	 * @return new dataset.
	 */
	public MIMLInstances insertAttributeToBags(Attribute newAttr) throws InvalidDataFormatException {

		// Obtains a copy of the instances
		Instances instances = new Instances(this.getDataSet());

		// Prepares new relational attribute with the current attributes
		Instances bag = instances.instance(0).relationalValue(1);
		ArrayList<Attribute> relationalAttributes = new ArrayList<Attribute>();
		for (int j = 0; j < bag.numAttributes(); j++)
			relationalAttributes.add(bag.attribute(j));

		// Inserts attribute as the last relational attribute
		relationalAttributes.add(newAttr);

		// Template for the new relational attribute
		Instances header = new Instances("header", relationalAttributes, 0);

		// Deletes the old relational attribute and adds the new one
		// This sets the values of the whole relational attribute to '?' (any instance)
		instances.deleteAttributeAt(1);
		instances.insertAttributeAt(new Attribute("bag", header), 1);

		// Inserts instances into bags
		for (int i = 0; i < this.getNumInstances(); i++) {

			// Gets the current bag
			bag = this.getDataSet().instance(i).relationalValue(1);

			// Generates the new bag
			Instances bagDest = new Instances("header", relationalAttributes, 0);
			for (int j = 0; j < bag.numInstances(); j++) {
				Instance aux = bag.instance(j);
				aux.setDataset(null);
				aux.insertAttributeAt(aux.numAttributes());
				bagDest.add(aux);
			}

			// Sets a new value for relational attribute
			int index = instances.instance(i).attribute(1).addRelation(bagDest);
			instances.instance(i).setValue(1, index);
		}

		return new MIMLInstances(instances, this.getLabelsMetaData());
	}

	/**
	 * Adds a set of attributes to the relational attribute with values '?' at the
	 * last position of the relational attribute.
	 * 
	 * @param Attributes ArrayList of attributes to add.
	 * @throws InvalidDataFormatException if occurred an error creating new dataset.
	 * 
	 * @return new dataset.
	 */
	public MIMLInstances insertAttributesToBags(ArrayList<Attribute> Attributes) throws InvalidDataFormatException {

		// Obtains a copy of the instances
		Instances instances = new Instances(this.getDataSet());

		// Prepares new relational attribute with the current attributes
		Instances bag = instances.instance(0).relationalValue(1);
		ArrayList<Attribute> relationalAttributes = new ArrayList<Attribute>();
		for (int j = 0; j < bag.numAttributes(); j++)
			relationalAttributes.add(bag.attribute(j));

		// Inserts attributes as the last relational attributes
		for (int j = 0; j < Attributes.size(); j++)
			relationalAttributes.add(Attributes.get(j));

		// Template for the new relational attribute
		Instances header = new Instances("header", relationalAttributes, 0);

		// Deletes the old relational attribute and adds the new one
		// This sets the values of the whole relational attribute to '?' (any instance)
		instances.deleteAttributeAt(1);
		instances.insertAttributeAt(new Attribute("bag", header), 1);

		// Inserts instances into bags
		for (int i = 0; i < this.getNumInstances(); i++) {

			// Gets the current bag
			bag = this.getDataSet().instance(i).relationalValue(1);

			// Generates the new bag
			Instances bagDest = new Instances("header", relationalAttributes, 0);
			for (int j = 0; j < bag.numInstances(); j++) {
				Instance aux = bag.instance(j);
				aux.setDataset(null);
				for (int k = 0; k < Attributes.size(); k++) {
					aux.insertAttributeAt(aux.numAttributes());
				}
				bagDest.add(aux);
			}

			// Sets a new value for relational attribute
			int index = instances.instance(i).attribute(1).addRelation(bagDest);
			instances.instance(i).setValue(1, index);

		}
		return new MIMLInstances(instances, this.getLabelsMetaData());
	}

	/**
	 * Split MIML data train and test partition given a percentage and a
	 * partitioning method.
	 *
	 * @param mimlDataSet        The MIML dataset to be splited.
	 * @param percentageTrain    The percentage (0-100) to be used for train.
	 * @param seed               Seed use to randomize.
	 * @param partitioningMethod An integer with the partitioning method:
	 *                           <ul>
	 *                           <li>1 random partitioning</li>
	 *                           <li>2 powerset partitioning</li>
	 *                           <li>3 iterative partitioning</li>
	 *                           </ul>
	 * @return A list with the dataset splited.
	 * @throws Exception To be handled in an upper level.
	 */
	public static MIMLInstances[] splitData(MIMLInstances mimlDataSet, double percentageTrain, int seed,
			int partitioningMethod) throws Exception {
		// splits the data set into train and test
		// copy of original data
		MultiLabelInstances[] partitionsML = null;

		switch (partitioningMethod) {
		case 1:
			RandomTrainTest engineRandom = new RandomTrainTest(seed, mimlDataSet);
			partitionsML = engineRandom.split(percentageTrain);
			break;
		case 2:
			LabelPowersetTrainTest engineLP = new LabelPowersetTrainTest(seed, mimlDataSet);
			partitionsML = engineLP.split(percentageTrain);
			break;
		case 3:
			IterativeTrainTest engineIterative = new IterativeTrainTest(seed, mimlDataSet);
			partitionsML = engineIterative.split(percentageTrain);
			break;
		}

		MIMLInstances[] partitions =  new MIMLInstances[2];
		partitions[0] = new MIMLInstances(partitionsML[0].getDataSet(), partitionsML[0].getLabelsMetaData()); 
		partitions[1] = new MIMLInstances(partitionsML[1].getDataSet(), partitionsML[1].getLabelsMetaData());

		return partitions;
	}
	
	
	/**
	 * Generate tran/test partitions for CV cross validation.
	 *
	 * @param mimlDataSet        The MIML dataset to be splited.
	 * @param nFolds             The number of folds.
	 * @param seed               Seed use to randomize.
	 * @param partitioningMethod An integer with the partitioning method:
	 *                           <ul>
	 *                           <li>1 random partitioning</li>
	 *                           <li>2 powerset partitioning</li>
	 *                           <li>3 iterative partitioning</li>
	 *                           </ul>
	 * @return MIMLInstances[][] a nfolds x 2 matrix. Each row represents a
	 *         fold being column 0 the train set and the column 1 the test set.
	 * @throws Exception To be handled in an upper level.
	 */
	public static MIMLInstances[][] roundsCV(MIMLInstances mimlDataSet, int nFolds, int seed,
			int partitioningMethod) throws Exception {
		
		MultiLabelInstances[][] roundsML = null;
		MIMLInstances[][] rounds = null;
		CrossValidationBase engine;
		switch (partitioningMethod) {
		case 1:
			engine = new RandomCrossValidation(mimlDataSet);
			roundsML = engine.getRounds(nFolds);
			break;
		case 2:
			engine = new LabelPowersetCrossValidation(mimlDataSet);
			roundsML = engine.getRounds(nFolds);
			break;
		case 3:
			engine = new IterativeCrossValidation(mimlDataSet);
			roundsML = engine.getRounds(nFolds);
			break;
		}

		rounds = new MIMLInstances[nFolds][2];
		for (int i=0; i<nFolds; i++)
		{
			rounds[i][0] = new MIMLInstances(roundsML[i][0].getDataSet(), roundsML[i][0].getLabelsMetaData());  //train set for round i
			rounds[i][1] = new MIMLInstances(roundsML[i][1].getDataSet(), roundsML[i][1].getLabelsMetaData());; //test set for round i
		}	
		return (rounds);
	}

}
