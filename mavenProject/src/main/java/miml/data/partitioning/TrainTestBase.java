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

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;

/**
 * General scheme for train test partitioning of multi-output data. MOR, MIML and MVML
 * formats are also supported.
 * 
 * @author Eva Gibaja
 * @version 20201029
 */
public abstract class TrainTestBase extends PartitionerBase {

	/**
	 * Constructor.
	 * 
	 * @param seed
	 *            Seed for randomization
	 * @param mlDataSet
	 *            A multi-label dataset
	 * @throws InvalidDataFormatException
	 *             To be handled
	 */
	public TrainTestBase(int seed, MultiLabelInstances mlDataSet) throws InvalidDataFormatException {
		super(seed, mlDataSet);
	}

	/**
	 * Default constructor.
	 * 
	 * @param mlDataSet
	 *            A multi-label dataset
	 * @throws InvalidDataFormatException
	 *             To be handled
	 */
	public TrainTestBase(MultiLabelInstances mlDataSet) throws InvalidDataFormatException {
		super(mlDataSet);
	}

	/**
	 * Returns a array with two multi-label random datasets corresponding to the
	 * train and test sets respectively.
	 *
	 * @param percentageTrain
	 *            Percentage of train dataset.
	 * @return MultiLabelInstances[].<br>
	 *         MultiLabelInstances[0] is the train set. <br>
	 *         MultiLabelInstances[1] is the test set.
	 * @throws java.lang.Exception
	 *             To be handled.
	 */
	public abstract MultiLabelInstances[] split(double percentageTrain) throws Exception;

}
