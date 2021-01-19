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

package miml.data.partitioning.random;

import java.util.ArrayList;
import java.util.Random;

import miml.data.partitioning.CrossValidationBase;
import weka.core.Instance;
import weka.core.Instances;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;

/**
 * Class to split a multi-label dataset into N multi-label random datasets for
 * cross-validation. MIML and MVML formats are also supported. Due to this fact,
 * applied over datasets with a high number of labels (e.g. some subsets of miml
 * protein data), this method may generate folds with uneven number of instances
 * and with some duplicated instances. In these cases, using a lower number of
 * folds (eg. 3 folds) or another kind of partitioning (eg. iteratrive or
 * powerset) is recommended. Besides, the same instance could be included twice
 * to guarantee instances of all labels in the resulte train set.
 * 
 * @author Eva Gibaja
 * @version 20201029
 */
public class RandomCrossValidation extends CrossValidationBase {

	/**
	 * A matrix of nFoldsx2 representing the index of the first and last instance of
	 * each partition
	 */
	protected int indexes[][] = null;

	/**
	 * Constructor.
	 * 
	 * @param seed
	 *            Seed for randomization
	 * @param mlDataSet
	 *            A multi-label dataset
	 * @throws InvalidDataFormatException
	 *             To be handled.
	 */
	public RandomCrossValidation(int seed, MultiLabelInstances mlDataSet) throws InvalidDataFormatException {
		super(seed, mlDataSet);
	}

	/**
	 * Default constructor.
	 * 
	 * @param mlDataSet
	 *            A multi-label dataset
	 * @throws InvalidDataFormatException
	 *             To be handled.
	 */
	public RandomCrossValidation(MultiLabelInstances mlDataSet) throws InvalidDataFormatException {
		super(mlDataSet);
	}

	@Override
	public MultiLabelInstances[] getFolds(int nFolds) throws InvalidDataFormatException {

		// copy of original data
		Instances dataSet = new Instances(workingSet.getDataSet());

		// randomize dataset
		dataSet.randomize(new Random(seed));

		// Initializations
		int numInstances = workingSet.getNumInstances();
		int numLabels = workingSet.getNumLabels();
		int labelIndices[] = workingSet.getLabelIndices();

		if (nFolds < 2) {
			throw new IllegalArgumentException("Number of folds must be at least 2!");
		}
		if (nFolds > workingSet.getNumInstances()) {
			throw new IllegalArgumentException("Can't have more folds than instances!");
		}

		// Having representative instances of each label in at least two folds, it would
		// be representatives for each label in each possible round. Otherwise reading
		// data would fail.
		// Fold0 and Fold1 will have these representatives

		// Instance indices of initial assignment. Initially -1
		int instancesAssignments[][] = new int[2][numLabels];
		for (int i = 0; i < 2; i++)
			for (int j = 0; j < numLabels; j++)
				instancesAssignments[i][j] = -1;

		int labelsCovered = 0;
		for (int i = 0; (i < numInstances) && (labelsCovered < 2 * numLabels); i++) {

			Instance instance = dataSet.instance(i);
			int selected_fold = -1; // instance not assigned initially to any fold

			for (int l = 0; (l < labelIndices.length) && (selected_fold < 0); l++) {
				if (instance.stringValue(labelIndices[l]).equals("1")) {
					if (instancesAssignments[0][l] == -1) {
						selected_fold = 0;
					} else if (instancesAssignments[1][l] == -1) {
						selected_fold = 1;
					}
				}
			}

			// Instance i has been initially selected
			if (selected_fold >= 0) {
				// All instance's labels not previously included are covered
				for (int l = 0; l < labelIndices.length; l++) {
					if (instance.stringValue(labelIndices[l]).equals("1")
							&& (instancesAssignments[selected_fold][l] == -1)) {
						instancesAssignments[selected_fold][l] = i;
						labelsCovered++;
					}
				}
			}
		} // for instances

		// If any label is only represented by an instance, the instance is included in
		// the two folds
		for (int f = 0, f2 = 1; f < 2; f++, f2--) {
			for (int l = 0; l < instancesAssignments[f].length; l++) {
				if (instancesAssignments[f][l] == -1) {

					int index = instancesAssignments[f2][l];
					Instance instance = dataSet.instance(index);

					// All labels not previously included are considered
					for (int k = 0; k < labelIndices.length; k++) {
						if ((instance.stringValue(labelIndices[k]).equals("1")) & (instancesAssignments[f][k] == -1)) {
							instancesAssignments[f][k] = index;
							labelsCovered++;
						}
					}
				}
			}
		}

		// All partitions will have the relation name of the original one.
		// This allows multi-view definitions of data stored in @relation to be
		// maintained.
		String relationName = workingSet.getDataSet().relationName();
		MultiLabelInstances Folds[] = new MultiLabelInstances[nFolds];
		for (int f = 0; f < nFolds; f++) {
			Folds[f] = new MultiLabelInstances(new Instances(dataSet, 0), workingSet.getLabelsMetaData());
			Folds[f].getDataSet().setRelationName(relationName);
		}

		// Instance indexes to assing to any fold
		ArrayList<Integer> toAssign = new ArrayList<Integer>();
		for (int i = 0; i < numInstances; i++)
			toAssign.add(i);
		int totalAssigned = 0;
		int indextoAssign = 0;

		// for further checking
		int choosen[] = new int[numInstances];
		for (int i = 0; i < numInstances; i++)
			choosen[i] = -1;

		// Adds initial instances to the two folds
		for (int f = 0; f < 2; f++) {

			ArrayList<Integer> list = new ArrayList<Integer>();
			for (int l = 0; l < numLabels; l++) {
				int instanceIndex = instancesAssignments[f][l];

				// to avoid insert the same instance twice
				if (list.contains(instanceIndex) == false) {

					Folds[f].getDataSet().add(dataSet.instance(instanceIndex));
					list.add(instanceIndex);
					toAssign.remove(instanceIndex);
					choosen[instanceIndex] = f;
					totalAssigned++;
				}
			}
		}

		// In turns, adds the rest of instances
		int patternsPerFold = (totalAssigned + toAssign.size()) / nFolds;
		for (int f = 0; indextoAssign < toAssign.size(); f = (f + 1) % nFolds) {
			// System.out.println("toAssign.size: "+toAssign.size()+" indexToAssign:
			// "+indextoAssign+" f: "+f+" size[f]:"+Folds[f].getNumInstances());
			// patternsPerFold+1 due to the offset
			if (Folds[f].getNumInstances() < (patternsPerFold + 1)) {
				int instanceIndex = toAssign.get(indextoAssign);
				Folds[f].getDataSet().add(dataSet.instance(instanceIndex));
				choosen[instanceIndex] = f;
				totalAssigned++;
				indextoAssign++;
			}
		}

		// Checking
		System.out.println("currentInstances/realInstances (may be duplicated): " + totalAssigned + "/" + numInstances);
		for (int f = 0; f < nFolds; f++)
			System.out.println("Fold " + f + ": " + Folds[f].getDataSet().numInstances() + " patterns");
		/*
		 * System.out.print("["); for (int i = 0; i < choosen.length; i++) { if
		 * (choosen[i] == -1) System.out.println("Instance " + i + " not choosen"); else
		 * System.out.print(choosen[i] + ","); } System.out.println("]");
		 */
		return Folds;
	}
}
