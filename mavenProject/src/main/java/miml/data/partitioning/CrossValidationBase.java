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
import weka.core.Instances;


/**
 * General scheme for cross validation partitioners of multi-output data. MOR, MIML
 * and MVML formats are also supported.
 * 
 * @author Eva Gibaja
 * @version 20201029
 */
public abstract class CrossValidationBase extends PartitionerBase {

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
	public CrossValidationBase(int seed, MultiLabelInstances mlDataSet) throws InvalidDataFormatException {
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
	public CrossValidationBase(MultiLabelInstances mlDataSet) throws InvalidDataFormatException {
		super(mlDataSet);
	}

	/**
	 * Returns the train and test sets for each fold.
	 *
	 * @param nFolds
	 *            Number of folds.
	 * @return MultiLabelInstances[][] a nfolds x 2 matrix. Each row represents a
	 *         fold being column 0 the train set and the column 1 the test set.
	 * @throws mulan.data.InvalidDataFormatException
	 *             To be handled.
	 */
	public MultiLabelInstances[][] getRounds(int nFolds) throws Exception {
		MultiLabelInstances Folds[] = getFolds(nFolds);
		return foldsToRounds(Folds);
	}

	/**
	 * Returns the train and test sets for each fold. This method is static being
	 * useful if the user has partitions.
	 *
	 * @param Folds
	 *            The folds.
	 * @return MultiLabelInstances[][] a nfolds x 2 matrix. Each row represents a
	 *         fold being column 0 the train set and the column 1 the test set.
	 * @throws Exception 
	 */
	public static MultiLabelInstances[][] foldsToRounds(MultiLabelInstances Folds[]) throws Exception {
		int nFolds = Folds.length;
		MultiLabelInstances Partition[][] = new MultiLabelInstances[nFolds][2];

		// All partitions will have the relation name of the original one.
		// This allows multi-view definitions of data stored in @relation to be
		// maintained.
		String relationName = Folds[0].getDataSet().relationName();		
		
		for (int i = 0; i < nFolds; i++) {
			// Prepares test partition
			Instances test  =  new Instances(Folds[i].getDataSet());
			test.addAll(new Instances(Folds[i].getDataSet()));			
			
			// Prepares train partition
			Instances train =  new Instances(Folds[i].getDataSet());
			for (int j = 0; j < nFolds; j++) {
				if (j != i)					
						train.addAll(new Instances(Folds[j].getDataSet()));				
			}

			Partition[i][0] =new MultiLabelInstances(new Instances(train), Folds[i].getLabelsMetaData());
			Partition[i][0].getDataSet().setRelationName(relationName);

			Partition[i][1] =new MultiLabelInstances(new Instances(test), Folds[i].getLabelsMetaData());
			Partition[i][1].getDataSet().setRelationName(relationName);	
		}			
		return Partition;
	}

	/**
	 * Splits a dataset into nfolds partitions.
	 *
	 * @param nFolds
	 *            Number of folds.
	 * @return MultiLabelInstances[] a vector of nFolds. Each element represents a
	 *         fold.
	 * @throws mulan.data.InvalidDataFormatException
	 *             To be handled.
	 */
	public abstract MultiLabelInstances[] getFolds(int nFolds) throws InvalidDataFormatException;

}
