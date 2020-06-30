/*    This program is free software; you can redistribute it and/or modify
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

package miml.classifiers.miml.meta;

import java.util.Arrays;

import org.apache.commons.configuration2.Configuration;

import miml.classifiers.miml.IMIMLClassifier;
import miml.classifiers.miml.MIMLClassifier;
import miml.core.IConfiguration;
import miml.core.Utils;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import weka.core.Instances;

/**
 * <p>
 * Class implementing an ensemble algorithm using bagging. For more information,
 * see <em>Breiman, L. (1996). Bagging predictors. Machine learning, 24(2),
 * 123-140.</em>
 * </p>
 * 
 * @author Alvaro A. Belmonte
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20180717
 */
public class MIMLBagging extends MIMLClassifier {

	/** Generated Serial version UID. */
	private static final long serialVersionUID = 1L;

	/** Threshold for predictions. */
	protected double threshold = 0.5;

	/** Seed for randomization. */
	protected int seed = 1;

	/**
	 * Determines whether the classifier will consider sampling with replacement. By
	 * default it is false.
	 */
	boolean sampleWithReplacement = true;

	/**
	 * Determines whether confidences [0,1] or relevance {0,1} is used to compute
	 * bipartition.
	 */
	boolean useConfidences = false;

	/** The size of the sample. */
	int samplePercentage = 100;

	/**
	 * Number of classifiers in the ensemble.
	 */
	protected int numClassifiers = 5;

	/** Base learner. */
	protected IMIMLClassifier baseLearner;

	/**
	 * The ensemble of MultiLabelLearners. To be initialized by the builder method.
	 */
	protected IMIMLClassifier ensemble[] = null;

	/**
	 * No-argument constructor for xml configuration.
	 */
	public MIMLBagging() {
	}

	/**
	 * Constructor of the class. Its default setting is: @li
	 * sampleWithReplacement=false @li threshold=0.5.
	 *
	 * @param baseLearner    The base learner to be used.
	 * @param numClassifiers The number of base classifiers in the ensemble.
	 */
	public MIMLBagging(IMIMLClassifier baseLearner, int numClassifiers) {
		this.baseLearner = baseLearner;
		this.numClassifiers = numClassifiers;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see mimlclassifier.MIMLClassifier#buildInternal(data.MIMLInstances)
	 */
	@Override
	protected void buildInternal(MIMLInstances trainingSet) throws Exception {

		// Prepares the ensemble
		ensemble = new MIMLClassifier[numClassifiers];

		for (int i = 0; i < numClassifiers; i++) {
			ensemble[i] = baseLearner.makeCopy();
			Instances sample = Utils.resample(trainingSet.getDataSet(), samplePercentage, sampleWithReplacement, seed + i);

			System.out.println("\t\tBase Classifier " + i + ": " + sample.numInstances() + "/"
					+ trainingSet.getNumBags() + " bags");
			ensemble[i].build(new MIMLInstances(sample, trainingSet.getLabelsMetaData()));
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see mimlclassifier.MIMLClassifier#makePredictionInternal(data.Bag)
	 */
	@Override
	protected MultiLabelOutput makePredictionInternal(MIMLBag instance) throws Exception, InvalidDataException {

		boolean bipartitions[][] = new boolean[ensemble.length][];
		double confidences[][] = new double[ensemble.length][];
		for (int i = 0; i < this.ensemble.length; i++) {

			MultiLabelOutput prediction = ensemble[i].makePrediction(instance);

			bipartitions[i] = prediction.getBipartition();
			confidences[i] = prediction.getConfidences();
		}

		double[] sumVotes = new double[numLabels]; // double to consider weights
		double[] sumConf = new double[numLabels];

		Arrays.fill(sumVotes, 0);
		Arrays.fill(sumConf, 0);

		for (int i = 0; i < numClassifiers; i++) {
			for (int j = 0; j < numLabels; j++) {
				sumVotes[j] += bipartitions[i][j] == true ? 1 : 0;
				sumConf[j] += confidences[i][j];
			}
		}

		double[] confidence = new double[numLabels];
		for (int j = 0; j < numLabels; j++) {
			if (useConfidences) {
				confidence[j] = sumConf[j] / numClassifiers;
			} else {
				confidence[j] = sumVotes[j] / numClassifiers;
			}
		}

		MultiLabelOutput mlo = new MultiLabelOutput(confidence, this.threshold);
		return mlo;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * core.IConfiguration#configure(org.apache.commons.configuration.Configuration)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void configure(Configuration configuration) {

		this.threshold = configuration.getDouble("threshold", 0.5);
		this.seed = configuration.getInt("seed", 1);

		this.sampleWithReplacement = configuration.getBoolean("sampleWithReplacement", true);
		this.useConfidences = configuration.getBoolean("useConfidences", false);

		this.samplePercentage = configuration.getInt("samplePercentage", 90);
		this.numClassifiers = configuration.getInt("numClassifiers", 5);

		try {
			// Get the base classifier name
			String clsName = configuration.getString("baseLearner[@name]");
			// Instance class
			Class<? extends IMIMLClassifier> clsClass = (Class<? extends IMIMLClassifier>) Class.forName(clsName);

			this.baseLearner = clsClass.newInstance();
			// Configure the classifier
			if (this.baseLearner instanceof MIMLClassifier)
				((IConfiguration) this.baseLearner).configure(configuration.subset("baseLearner"));
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}

	}

	/**
	 * Sets the seed value.
	 *
	 * @param seed The seed value.
	 */
	public void setSeed(int seed) {
		this.seed = seed;
	}

	/**
	 * Returns the number of classifiers of the ensemble.
	 *
	 * @return Number of classifiers.
	 */
	public int getNumClassifiers() {
		return numClassifiers;
	}

	/**
	 * Returns the percentage of instances used for sampling with replacement.
	 *
	 * @return The sample percentage.
	 */
	public int getSamplePercentage() {
		return samplePercentage;
	}

	/**
	 * Sets the percentage of instances used for sampling with replacement*.
	 *
	 * @param samplePercentage The size of the sample referring the original one.
	 */

	public void setSamplePercentage(int samplePercentage) {
		this.samplePercentage = samplePercentage;
	}

	/**
	 * Returns true if the algorithm is configured with sampling and false
	 * otherwise.
	 *
	 * @return True if the algorithm is configured with sampling and false otherwise.
	 */
	public boolean isSampleWithReplacement() {
		return sampleWithReplacement;
	}

	/**
	 * Configure the classifier to use/not use sampling with replacement.
	 *
	 * @param sampleWithReplacement True if the classifier is set to use sampling
	 *                              with replacement.
	 */
	public void setSampleWithReplacement(boolean sampleWithReplacement) {
		this.sampleWithReplacement = sampleWithReplacement;
	}

	/**
	 * Returns the value of the threshold.
	 *
	 * @return double The threshold.
	 */
	public double getThreshold() {
		return threshold;
	}

	/**
	 * Sets the value of the threshold.
	 *
	 * @param threshold The value of the threshold.
	 */
	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}

	/**
	 * Returns whether the classifier uses confidences of bipartitions to combine
	 * classifiers in the ensemble.
	 *
	 * @return True, if is use confidences.
	 */

	public boolean isUseConfidences() {
		return useConfidences;
	}

	/**
	 * Stablishes whether confidences or bipartitions are used to combine classifiers
	 * in the ensemble.
	 *
	 * @param useConfidences The value of the property.
	 */

	public void setUseConfidences(boolean useConfidences) {
		this.useConfidences = useConfidences;
	}
}
