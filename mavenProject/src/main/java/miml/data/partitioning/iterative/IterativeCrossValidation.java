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

/*
 *    IterativeCrossValidation.java
 *    This java class is based on the mulan.data.IterativeStratification.java 
 *    class provided in the mulan java framework for multi-label learning
 *    Tsoumakas, G., Katakis, I., Vlahavas, I. (2010) "Mining Multi-label Data", 
 *    Data Mining and Knowledge Discovery Handbook, O. Maimon, L. Rokach (Ed.),
 *    Springer, 2nd edition, 2010.
 */

package miml.data.partitioning.iterative;

import miml.data.partitioning.CrossValidationBase;
import mulan.data.InvalidDataFormatException;
import mulan.data.IterativeStratification;
import mulan.data.MultiLabelInstances;
import weka.core.Instances;

/**
 * Class to carry out an stratified cross validation partition of multi-label
 * dataset. MIML and MVML format is also supported.
 * 
 * This java class is based on the mulan.data.IterativeStratification.java class
 * provided in the Mulan java framework for multi-label learning Tsoumakas, G.,
 * Katakis, I., Vlahavas, I. (2010) "Mining Multi-label Data", Data Mining and
 * Knowledge Discovery Handbook, O. Maimon, L. Rokach (Ed.), Springer, 2nd
 * edition, 2010. The method is described in Sechidis, K.; Tsoumakas, G. and
 * Vlahavas, I. Gunopulos, D.; Hofmann, T.; Malerba, D. and Vazirgiannis, M.
 * (Eds.) On the Stratification of Multi-label Data Machine Learning and
 * Knowledge Discovery in Databases, Springer Berlin Heidelberg, 2011, 6913,
 * 145-158. Our contribution is the adaptation of method split to generate
 * train-test partition.
 * 
 * @author Eva Gibaja
 * @version 20201029
 */

public class IterativeCrossValidation extends CrossValidationBase {

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
	public IterativeCrossValidation(int seed, MultiLabelInstances mlDataSet) throws InvalidDataFormatException {
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
	public IterativeCrossValidation(MultiLabelInstances mlDataSet) throws InvalidDataFormatException {
		super(mlDataSet);
	}

	@Override
	public MultiLabelInstances[] getFolds(int nFolds) throws InvalidDataFormatException {

		IterativeStratification engine = new IterativeStratification(seed);

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
