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
package miml.classifiers.miml.neural;

import org.apache.commons.configuration2.Configuration;

import com.mathworks.toolbox.javabuilder.MWCellArray;
import com.mathworks.toolbox.javabuilder.MWClassID;
import com.mathworks.toolbox.javabuilder.MWException;
import com.mathworks.toolbox.javabuilder.MWNumericArray;
import lamda.EnMIMLNNmetric;

import miml.classifiers.miml.MIMLClassifier;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import miml.data.MWWrapper;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;

/**
 * <p>
 * Class to execute the
 * <a href="http://www.lamda.nju.edu.cn/code_EnMIMLNNmetric.ashx">EnMIMLNNmetric
 * algorithm</a> for MIML data. For more information, see <em>Wu, J. S., Huang,
 * S. J., &amp; Zhou, Z. H. (2014). Genome-wide protein function prediction through
 * multi-instance multi-label learning. IEEE/ACM Transactions on Computational
 * Biology and Bioinformatics, 11(5), 891-902.</em>.
 * </p>
 * 
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20211027
 */
public class EnMIMLNNmetric_MIMLWrapper extends MIMLClassifier {

	/** For serialization. */
	private static final long serialVersionUID = -8995828753405862133L;

	/** A matlab object wrapping the EnMIMLNNmetric algorithm. */
	static EnMIMLNNmetric enmimlnn;

	/** Wrapper for Matlab data types. */
	protected MWWrapper wrapper;

	// Default parameters: ratio=0.1, mu=0.8 seed=1

	/** The number of centroids of the i-th label is set to be ratio*Ti, where Ti is the number of train bags with label i. */
	double ratio = 0.1;

	/** The ratio used to determine the standard deviation of the Gaussian activation function. */
	double mu = 0.8;

	/** Seed for kmedoids clustering. */
	int seed = 1;

	/**
	 * It will store the trained classifier being
	 * <ul>
	 * <li>classifier[0] the centroids. A Kx1 cell structure, where the k-th
	 * centroid of the RBF neural network is stored in Centroid{k,1}</li>
	 * <li>classifier[1] the sigma value. A 1xK vector, where the sigma value for
	 * the k-th centroid is stored in Sigma_value(1,k)</li>
	 * <li>classifier[2] the weights. A (K+1)xQ matrix used for label
	 * prediction</li>
	 * </ul>
	 */
	Object[] classifier = null;

	/**
	 * No-argument constructor for xml configuration.
	 * 
	 * @throws MWException To be handled.
	 */
	public EnMIMLNNmetric_MIMLWrapper() throws MWException {		
		enmimlnn = new EnMIMLNNmetric();
	}

	/**
	 * Basic constructor to initialize the classifier.
	 * 
	 * @param ratio The fraction parameter of EnMIMLNNmetric.
	 * @param mu    The scaling factor of EnMIMLNNmetric.
	 * @throws MWException To be handled. 
	 */
	public EnMIMLNNmetric_MIMLWrapper(double ratio, double mu) throws MWException {
		this();
		this.ratio = ratio;
		this.mu = mu;
	}

	/**
	 * Constructor to initialize the classifier.
	 * 
	 * @param ratio The fraction parameter of EnMIMLNNmetric.
	 * @param mu    The scaling factor of EnMIMLNNmetric.
	 * @param seed  Seed for kmedoids clustering.
	 * @throws MWException To be handled.
	 */
	public EnMIMLNNmetric_MIMLWrapper(double ratio, double mu, int seed) throws MWException {
		this(ratio, mu);
		this.seed = seed;
	}

	/**
	 * Returns the fraction parameter considered to build the classifier.
	 * @return The fraction parameter considered to build the classifier.
	 */
	public double getRatio() {
		return ratio;
	}

	/**
	 * Sets the fraction parameter to build the classifier.
	 * 
	 * @param ratio The fraction parameter of EnMIMLNNmetric.
	 */
	public void setRatio(double ratio) {
		this.ratio = ratio;
	}

	/**
	 * Returns the scaling factor parameter considered to build the classifier.
	 * @return The scaling factor parameter considered to build the classifier.
	 */
	public double getMu() {
		return mu;
	}

	/**
	 * Sets the scaling factor parameter to build the classifier.
	 * 
	 * @param mu The scaling factor of EnMIMLNNmetric.
	 */
	public void setMu(double mu) {
		this.mu = mu;
	}

	/**
	 * Returns the seed for kmedoids clustering considered to build the classifier.
	 * 
	 * @return The seed for kmedoids clustering considered to build the classifier.
	 */
	public int getSeed() {
		return seed;
	}

	/**
	 * Sets the seed for kmedoids clustering considered to build the classifier.
	 * @param seed The seed
	 */
	public void setSeed(int seed) {
		this.seed = seed;
	}

	@Override
	public void configure(Configuration configuration) {
		this.ratio = configuration.getDouble("ratio", 0.1);
		this.mu = configuration.getDouble("mu", 0.8);
		this.seed = configuration.getInt("seed", 1);
	}

	@Override
	protected void buildInternal(MIMLInstances trainingSet) throws Exception {
		wrapper = new MWWrapper(trainingSet);

		MWCellArray train_bags = wrapper.getBags();
		MWNumericArray train_target = wrapper.getLabels();
		MWNumericArray ratioIn = new MWNumericArray(ratio, MWClassID.DOUBLE);
		MWNumericArray muIn = new MWNumericArray(mu, MWClassID.DOUBLE);
		MWNumericArray seedIn = new MWNumericArray(seed, MWClassID.DOUBLE);

		// Call in Matlab:
		// [Centroids,Sigma_value,Weights]=EnMIMLNNmetric_run_train(train_bags,train_target,ratio,mu,seed)
		// When returning values, the first parameter of _run method must be the number
		// of returned values
		int nValuesReturned = 3;
		classifier = enmimlnn.EnMIMLNNmetric_run_train(nValuesReturned, train_bags, train_target, ratioIn, muIn,
				seedIn);
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(MIMLBag aBag) throws Exception, InvalidDataException {
		MWNumericArray test_bag = wrapper.getBagAsArray(aBag);
		MWNumericArray test_target = wrapper.getLabels(aBag);

		// Call in Matlab:
		//[Outputs,Pre_Labels]=EnMIMLNNmetric_run_test(aBag, targets, Centroids, Sigma_value, Weights);   
		// When returning values, the first parameter of _run method must be the number
		// of returned values
		int nValuesReturned = 2;

		Object[] prediction = enmimlnn.EnMIMLNNmetric_run_test(nValuesReturned, test_bag, test_target, classifier[0],
				classifier[1], classifier[2]);
		// Object[0] is a nLabelsx1 array of double containing the probability of the
		// testing instance belonging to each label
		// Object[1] is a nLabelsx1 array of double containing a bipartition being 1 if
		// the label is relevant or -1 otherwise

		double outputs[] = null;
		double pre_labels[] = null;
		if (prediction[0] instanceof MWNumericArray) {
			outputs = ((MWNumericArray) prediction[0]).getDoubleData();
		}

		if (prediction[1] instanceof MWNumericArray) {
			pre_labels = ((MWNumericArray) prediction[1]).getDoubleData();
		}

		boolean[] bipartition = new boolean[numLabels];
		double[] confidences = new double[numLabels];
		for (int l = 0; l < numLabels; l++) {
			bipartition[l] = (pre_labels[l] < 0) ? false : true;
			confidences[l] = outputs[l];
		}

		MultiLabelOutput finalDecision = new MultiLabelOutput(bipartition, confidences);
		return finalDecision;
	}
}
