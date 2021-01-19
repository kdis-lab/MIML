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
package miml.data.partitioning.powerset;

import miml.data.partitioning.CrossValidationBase;
import mulan.data.InvalidDataFormatException;
import mulan.data.LabelPowersetStratification;
import mulan.data.MultiLabelInstances;
import weka.core.Instances;

/**
 * Class to split a multi-label dataset into N multi-label for cross-validation
 * by applying a labelPowerset-based partition. MIML and MVML formats are also
 * supported.
 * 
 * @author Eva Gibaja
 * @version 20201029
 */

public class LabelPowersetCrossValidation extends CrossValidationBase {

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
	public LabelPowersetCrossValidation(int seed, MultiLabelInstances mlDataSet) throws InvalidDataFormatException {
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
	public LabelPowersetCrossValidation(MultiLabelInstances mlDataSet) throws InvalidDataFormatException {
		super(mlDataSet);
	}

	@Override
	public MultiLabelInstances[] getFolds(int nFolds) throws InvalidDataFormatException {

		LabelPowersetStratification engine = new LabelPowersetStratification(seed);

		// Copy of the dataset to ensure rounds and folds being generated from
		// the same original data
		MultiLabelInstances data = new MultiLabelInstances(new Instances(workingSet.getDataSet()),
				workingSet.getLabelsMetaData());
		MultiLabelInstances[] partition = engine.stratify(data, nFolds);

		// All partitions will have the relation name of the original one.
		// This allows multi-view definitions of data stored in @relation to be
		// maintained.
		String relationName = workingSet.getDataSet().relationName();
		for (int i = 0; i < partition.length; i++)
			partition[i].getDataSet().setRelationName(relationName);

		return partition;
	}

}
