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

import miml.data.partitioning.TrainTestBase;
import weka.core.Instance;
import weka.core.Instances;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;

/**
 * Class to split a multi-label dataset into two multi-label random datasets
 * corresponding to the train and test datasets respectively. MIML and MVML
 * formats are also supported. This class guarantees at least one instance for
 * label in train dataset.
 * 
 * @author Eva Gibaja
 * @version 20201029
 */

public class RandomTrainTest extends TrainTestBase {

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
	public RandomTrainTest(int seed, MultiLabelInstances mlDataSet) throws InvalidDataFormatException {
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
	public RandomTrainTest(MultiLabelInstances mlDataSet) throws InvalidDataFormatException {
		super(mlDataSet);
	}

	@Override
	public MultiLabelInstances[] split(double percentageTrain) throws Exception {

		// copy of original data
		Instances dataSet = new Instances(workingSet.getDataSet());

		// randomize dataset
		dataSet.randomize(new Random(seed));

		// Initializations
		int numInstances = workingSet.getNumInstances();
		int numLabels = workingSet.getNumLabels();
		int labelIndices[] = workingSet.getLabelIndices();
		int nTrain = (int) ((percentageTrain / 100) * numInstances);

		// Represents whether an instance has been selected for train set
		boolean selectedTrain[] = new boolean[numInstances];
		for (int i = 0; i < selectedTrain.length; i++)
			selectedTrain[i] = false;

		// Labels to cover, initally all labels
		ArrayList<Integer> labelsToCover = new ArrayList<>();
		for (int i = 0; i < labelIndices.length; i++)
			labelsToCover.add(labelIndices[i]);
		int nCoveredLabels = 0;

		// Adss to train set patterns of all labels. Train set must have patterns of all
		// labels otherwise multi-label classifiers could fail
		for (int i = 0; (i < numInstances) && (nCoveredLabels < numLabels); i++) {

			Instance instance = dataSet.instance(i);

			for (int j = 0; j < labelsToCover.size(); j++) {
				int index = labelsToCover.get(j);

				if (instance.stringValue(index).equals("1")) {
					labelsToCover.remove(j);
					selectedTrain[i] = true;
					nCoveredLabels++;
					// In addition, considers all labels included by the instance
					for (int k = 0; k < labelsToCover.size(); k++) {
						index = labelsToCover.get(k);
						if (instance.stringValue(index).equals("1")) {
							labelsToCover.remove(j);
							nCoveredLabels++;
						}
					}
				}
			}
		}

		MultiLabelInstances Partition[] = new MultiLabelInstances[2];
		Partition[0] = new MultiLabelInstances(new Instances(dataSet, 0), workingSet.getLabelsMetaData());
		Partition[1] = new MultiLabelInstances(new Instances(dataSet, 0), workingSet.getLabelsMetaData());

		// All partitions will have the relation name of the original one.
		// This allows multi-view definitions of data stored in @relation to be
		// maintained.
		String relationName = workingSet.getDataSet().relationName();
		Partition[0].getDataSet().setRelationName(relationName);
		Partition[1].getDataSet().setRelationName(relationName);

		// 0 not selected, 1 initally for train, 2 selected for train, 3 selected for
		// test
		int check[] = new int[dataSet.numInstances()];

		// Ensures all labels are represented in train
		int count = 0;
		for (int i = 0; i < selectedTrain.length; i++) {
			if (selectedTrain[i]) {
				Partition[0].getDataSet().add(dataSet.instance(i));
				count++;
				check[i] = 1;
			}
		}

		// System.out.println("\nInitially selected "+count+" instances for train to
		// ensure all labels presence");
		for (int i = 0; i < selectedTrain.length; i++) {
			if (!selectedTrain[i]) {
				if (count < nTrain) {
					Partition[0].getDataSet().add(dataSet.instance(i));
					count++;
					check[i] = 2;
				} else {
					Partition[1].getDataSet().add(dataSet.instance(i));
					check[i] = 3;
				}
			}
		}

		System.out.println("\nnLabels: " + numLabels + " Covered: " + nCoveredLabels);
		int currentTrain = Partition[0].getNumInstances();
		int currentTest = Partition[1].getNumInstances();
		System.out.println("\nTotal instances: " + numInstances + " Expected train: " + nTrain);
		System.out.println("\tCurrent: Train: " + currentTrain + " Test: " + currentTest + " Total: "
				+ (currentTrain + currentTest));

		/*
		 * System.out.
		 * println("\nResult of partitioning (0 not selected, 1 initially selected for train, 2 selected for train, 3 selected for test): "
		 * );
		 * 
		 * for (int i = 0; i < check.length; i++) System.out.print(check[i] + " ");
		 * System.out.println("\n");
		 */
		return Partition;

	}

}
