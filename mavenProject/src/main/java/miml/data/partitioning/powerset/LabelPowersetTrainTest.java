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

import java.util.Random;

import miml.data.partitioning.TrainTestBase;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.transformations.LabelPowersetTransformation;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Add;

/**
 * Class to split a multi-label dataset into two multi-label datasets
 * corresponding to the train and test datasets respectively by applying a
 * labelPowerset-based partition. MIML and MVML formats are also supported.
 * 
 * @author Eva Gibaja
 * @version 20201029
 */
public class LabelPowersetTrainTest extends TrainTestBase {

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
	public LabelPowersetTrainTest(int seed, MultiLabelInstances mlDataSet) throws InvalidDataFormatException {
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
	public LabelPowersetTrainTest(MultiLabelInstances mlDataSet) throws InvalidDataFormatException {
		super(mlDataSet);
	}

	@Override
	public MultiLabelInstances[] split(double percentage) throws Exception {
		LabelPowersetTransformation transformation = new LabelPowersetTransformation();
		Instances transformed = transformation.transformInstances(workingSet);

		// add id
		Add add = new Add();
		add.setAttributeIndex("first");
		add.setAttributeName("instanceID");
		add.setInputFormat(transformed);
		transformed = Filter.useFilter(transformed, add);
		for (int i = 0; i < transformed.numInstances(); i++) {
			transformed.instance(i).setValue(0, i);
		}
		transformed.setClassIndex(transformed.numAttributes() - 1);

		// stratify
		transformed.randomize(new Random(seed));

		// Resample supervised filter creates a stratified subsample of the
		// given dataset
		Resample rsp = new Resample();
		rsp.setNoReplacement(true); // sampling without replacement
		rsp.setSampleSizePercent(percentage);
		rsp.setInputFormat(transformed);
		Instances temp = Filter.useFilter(transformed, rsp);

		// Generate train
		Instances trainDataSet = new Instances(workingSet.getDataSet(), 0);
		for (int j = 0; j < temp.numInstances(); j++) {
			trainDataSet.add(workingSet.getDataSet().instance((int) temp.instance(j).value(0)));
		}

		// Generate test
		rsp = new Resample();
		rsp.setNoReplacement(true);
		rsp.setSampleSizePercent(percentage);
		rsp.setInvertSelection(true);
		rsp.setInputFormat(transformed);
		temp = Filter.useFilter(transformed, rsp);
		Instances testDataSet = new Instances(workingSet.getDataSet(), 0);
		for (int j = 0; j < temp.numInstances(); j++) {
			testDataSet.add(workingSet.getDataSet().instance((int) temp.instance(j).value(0)));
		}

		MultiLabelInstances Partition[] = new MultiLabelInstances[2];
		Partition[0] = new MultiLabelInstances(trainDataSet, workingSet.getLabelsMetaData());
		Partition[1] = new MultiLabelInstances(testDataSet, workingSet.getLabelsMetaData());

		// All partitions will have the relation name of the original one.
		// This allows multi-view definitions of data stored in @relation to be
		// maintained.
		String relationName = workingSet.getDataSet().relationName();
		Partition[0].getDataSet().setRelationName(relationName);
		Partition[1].getDataSet().setRelationName(relationName);

		return Partition;

	}

}
