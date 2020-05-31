package miml.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemovePercentage;
import weka.filters.unsupervised.instance.Resample;

public final class Utils {

	/**
	 * Obtains a sample of the original data.
	 *
	 * @param data                  Instances with the dataset.
	 * @param percentage            percentage of instances that will contain the
	 *                              new dataset.
	 * @param sampleWithReplacement If true the sample will be with replacement.
	 * @param seed                  Seed for randomization. Necessary if instances
	 *                              have not been previously shuffled with
	 *                              randomize.
	 * 
	 * @return Instances.
	 * @throws Exception To be handled.
	 */
	public static Instances resample(Instances data, double percentage, boolean sampleWithReplacement, int seed)
			throws Exception {

		Instances resampled;

		Resample rspl = new Resample();
		rspl.setRandomSeed(seed);
		rspl.setSampleSizePercent(percentage);
		rspl.setNoReplacement(!sampleWithReplacement);
		rspl.setInputFormat(data);
		resampled = Filter.useFilter(data, rspl);

		return resampled;
	}

	/**
	 * Split data given a percentage.
	 *
	 * @param mimlDataSet     The MIML dataset to be splited.
	 * @param percentageTrain The percentage (0-100) to be used in train.
	 * @return A list with the dataset splited.
	 * @throws Exception To be handled in an upper level.
	 */
	public static List<MIMLInstances> splitData(MIMLInstances mimlDataSet, double percentageTrain, int seed) throws Exception {
		// splits the data set into train and test
		// copy of original data
		Instances dataSet = new Instances(mimlDataSet.getDataSet());
		dataSet.randomize(new Random(seed));

		// obtains train set
		RemovePercentage rmvp = new RemovePercentage();
		rmvp.setInvertSelection(true);
		rmvp.setPercentage(percentageTrain);
		rmvp.setInputFormat(dataSet);
		Instances trainDataSet = Filter.useFilter(dataSet, rmvp);

		// obtains test set
		rmvp = new RemovePercentage();
		rmvp.setPercentage(percentageTrain);
		rmvp.setInputFormat(dataSet);
		Instances testDataSet = Filter.useFilter(dataSet, rmvp);
		
		List<MIMLInstances> datasets = new ArrayList<MIMLInstances>();
		datasets.add(new MIMLInstances(trainDataSet, mimlDataSet.getLabelsMetaData()));
		datasets.add(new MIMLInstances(testDataSet, mimlDataSet.getLabelsMetaData()));

		return datasets;
	}

}
