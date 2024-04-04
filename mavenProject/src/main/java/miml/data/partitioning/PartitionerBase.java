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
package miml.data.partitioning;

import java.util.Random;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.core.Instances;

/**
 * General scheme for partitioning multi-output data.
 * 
 * @author Eva Gibaja
 * @version 20201029
 */
public abstract class PartitionerBase {

	/** Seed for reproduction of results */
	protected int seed = 1;

	/** A copy of the instances to generate partitions */
	protected MultiLabelInstances workingSet;

	/**
	 * Constructor of the class
	 * 
	 * @param mlDataSet The multi-label data set
	 * @throws InvalidDataFormatException To be handled.
	 */
	public PartitionerBase(MultiLabelInstances mlDataSet) throws InvalidDataFormatException {
		super();
		// copy of original data to ensure the same initial dataset
		this.workingSet = new MultiLabelInstances(new Instances(mlDataSet.getDataSet()),
				mlDataSet.getLabelsMetaData().clone());
		
		//randomize according to the seed
		this.workingSet.getDataSet().randomize(new Random(seed));
		
	}

	/**
	 * Constructor of the class
	 * 
	 * @param seed      Seed for randomization
	 * @param mlDataSet The multi-label data set
	 * @throws InvalidDataFormatException To be handled.
	 */
	public PartitionerBase(int seed, MultiLabelInstances mlDataSet) throws InvalidDataFormatException {
		this(mlDataSet);
		// set seed value
		this.seed = seed;
	}

	/**
	 * Returns the number of examples of the dataset to be partitioned.
	 * 
	 * @return int
	 */
	public int totalExamples() {		
		return workingSet.getDataSet().numInstances();
	}

	/**
	 * Given an array with datasets corresponding to partitions, prints the number
	 * of examples of each dataset of the vector
	 * 
	 * @param Partition An array with the partitions. In case of train/test,
	 *                  partition Partition[0] is the train set and Partition[1] is
	 *                  the test set. In case of CV, Partition[i] is the ith fold.
	 */
	protected abstract void statsToString(MultiLabelInstances[] Partition);

}