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

import lamda.MIMLNN;

import miml.classifiers.miml.MIMLClassifier;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import miml.data.MWWrapper;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;

/**
 * <p>
 * Class to execute the
 * <a href="http://www.lamda.nju.edu.cn/code_MIML.ashx">MIMLNN algorithm</a> for
 * MIML data. For more information, see <em>Zhou, Z. H., Zhang, M. L., Huang, S.
 * J., &amp; Li, Y. F. (2012). Multi-instance multi-label learning. Artificial
 * Intelligence, 176(1), 2291-2320.</em>.
 * </p>
 * 
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20211030
 */
public class MIMLNN_MIMLWrapper extends MIMLClassifier {

	/** For serialization. */
	private static final long serialVersionUID = 8779276219977866499L;

	/** A matlab object wrapping the EnMIMLNNmetric algorithm. */
	static MIMLNN mimlnn;

	/** Wrapper for Matlab data types. */
	protected MWWrapper wrapper;

	// Default parameters: ratio=0.4, lambda=1, seed=1
	/** The number of clusters is set to ratio*numberOfTrainingBags, default=0.4. */
	double ratio = 0.4;

	/** The regularization parameter used to compute matrix inverse, default=1. */
	double lambda = 1;

	/** The seed for kmedoids clustering */
	int seed = 1;

	/**
	 * It will store the trained classifier being
	 * <ul>
	 * <li>classifier[0] the centroids. A Kx1 cell structure, where the k-th
	 * centroid of the RBF neural network is stored in Centroid{k,1}</li>
	 * <li>classifier[1] the weights. A (K+1)xQ matrix used for label
	 * prediction</li>
	 * </ul>
	 */
	Object[] classifier = null;

	/**
	 * No-argument constructor for xml configuration.
	 * 
	 * @throws MWException To be handled.
	 */
	public MIMLNN_MIMLWrapper() throws MWException {
		super();
		mimlnn = new MIMLNN();
	}

	/**
	 * Basic constructor to initialize the classifier.
	 * 
	 * @param ratio  The number of clusters is set to ratio*numberOfTrainingBags.
	 * @param lambda The regularization parameter used to compute matrix inverse
	 * @throws MWException To be handled.
	 */
	public MIMLNN_MIMLWrapper(double ratio, double lambda) throws MWException {
		this();
		this.ratio = ratio;
		this.lambda = lambda;
	}

	/**
	 * Constructor to initialize the classifier.
	 * 
	 * @param ratio  TThe number of clusters is set to ratio*numberOfTrainingBags.
	 * @param lambda The regularization parameter used to compute matrix inverse
	 * @param seed   Seed for kmedoids clustering.
	 * @throws MWException To be handled.
	 */
	public MIMLNN_MIMLWrapper(double ratio, double lambda, int seed) throws MWException {
		this(ratio, lambda);
		this.seed = seed;
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
	 * 
	 * @param seed The seed
	 */
	public void setSeed(int seed) {
		this.seed = seed;
	}

	/**
	 * Returns the fraction parameter considered to determine the number of clusters
	 * to build the classifier.
	 * 
	 * @return The fraction parameter considered to determine the number of clusters
	 *         to build the classifier.
	 */
	public double getRatio() {
		return ratio;
	}

	/**
	 * Sets the fraction parameter considered to determine the number of clusters to
	 * build the classifier.
	 * 
	 * @param ratio The fraction parameter considered to determine the number of
	 *              clusters to build the classifier.
	 */
	public void setRatio(double ratio) {
		this.ratio = ratio;
	}

	/**
	 * Returns the regularization parameter used to compute matrix inverse.
	 * 
	 * @return The regularization parameter used to compute matrix inverse.
	 */
	public double getLambda() {
		return lambda;
	}

	/**
	 * Sets the fraction parameter considered to determine the number of clusters to
	 * build the classifier.
	 * 
	 * @param lambda The fraction parameter considered to determine the number of
	 *               clusters to build the classifier.
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	@Override
	protected void buildInternal(MIMLInstances trainingSet) throws Exception {
		wrapper = new MWWrapper(trainingSet);

		MWCellArray train_bags = wrapper.getBags();
		MWNumericArray train_target = wrapper.getLabels();
		MWNumericArray ratioIn = new MWNumericArray(ratio, MWClassID.DOUBLE);
		MWNumericArray lambdaIn = new MWNumericArray(lambda, MWClassID.DOUBLE);
		MWNumericArray seedIn = new MWNumericArray(seed, MWClassID.DOUBLE);

		// Call in Matlab:
		// [Centroids, Weights] = MIMLNN_run_train(train_bags,train_target,ratio,
		// lambda, seed);
		// When returning values, the first parameter of _run method must be the number
		// of returned values
		int nValuesReturned = 2;
		classifier = mimlnn.MIMLNN_run_train(nValuesReturned, train_bags, train_target, ratioIn, lambdaIn, seedIn);

	}

	@Override
	protected MultiLabelOutput makePredictionInternal(MIMLBag aBag) throws Exception, InvalidDataException {
		
		MWNumericArray test_bag = wrapper.getBagAsArray(aBag);
		MWNumericArray test_target = wrapper.getLabels(aBag);

		// Call in Matlab:
		//[Outputs,Pre_Labels] = MIMLNN_run_test(aBag, targets, Centroids, Weights);  
		// When returning values, the first parameter of _run method must be the number
		// of returned values
		int nValuesReturned = 2;

		Object[] prediction = mimlnn.MIMLNN_run_test(nValuesReturned, test_bag, test_target, classifier[0],
				classifier[1]);
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

	@Override
	public void configure(Configuration configuration) {
		this.ratio = configuration.getDouble("ratio", 0.4);
		this.lambda = configuration.getDouble("lamda", 1);
		this.seed = configuration.getInt("seed", 1);
	}
}
