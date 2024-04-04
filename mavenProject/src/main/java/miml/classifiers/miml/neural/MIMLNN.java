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

import MWAlgorithms.MWMIMLNN;
import miml.classifiers.miml.MWClassifier;

/**
 * <p>
 * Class to execute the
 * <a href="http://www.lamda.nju.edu.cn/code_MIML.ashx">MIMLNN</a>algorithm for
 * MIML data. For more information, see <em>Zhou, Z. H., Zhang, M. L., Huang, S.
 * J., &amp; Li, Y. F. (2012). Multi-instance multi-label learning. Artificial
 * Intelligence, 176(1), 2291-2320.</em>.
 * </p>
 * 
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20211030
 */
public class MIMLNN extends MWClassifier {

	/** For serialization. */
	private static final long serialVersionUID = 8779276219977866499L;

	/** A matlab object wrapping the EnMIMLNNmetric algorithm. */
	static MWMIMLNN mimlnn;

	// Default parameters: ratio=0.4, lambda=1, seed=1
	/** The number of clusters is set to ratio*numberOfTrainingBags, default=0.4. */
	double ratio = 0.4;

	/** The regularization parameter used to compute matrix inverse, default=1. */
	double lambda = 1;

	/** The seed for kmedoids clustering */
	int seed = 1;

	/**
	 * No-argument constructor for xml configuration.
	 * 
	 * @throws MWException To be handled.
	 */
	public MIMLNN() throws MWException {
		super();
		mimlnn = new MWMIMLNN();
	}

	/**
	 * Basic constructor to initialize the classifier.
	 * 
	 * @param ratio  The number of clusters is set to ratio*numberOfTrainingBags.
	 * @param lambda The regularization parameter used to compute matrix inverse
	 * @throws MWException To be handled.
	 */
	public MIMLNN(double ratio, double lambda) throws MWException {
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
	public MIMLNN(double ratio, double lambda, int seed) throws MWException {
		this(ratio, lambda);
		this.seed = seed;
	}

	@Override
	public void dispose() {
		// Dispose of native MW resources
		mimlnn.dispose();
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
	public void configure(Configuration configuration) {
		this.ratio = configuration.getDouble("ratio", 0.4);
		this.lambda = configuration.getDouble("lamda", 1);
		this.seed = configuration.getInt("seed", 1);
	}

	@Override
	protected void trainMWClassifier(MWCellArray train_bags, MWNumericArray train_targets) throws MWException {

		MWNumericArray ratioIn = new MWNumericArray(ratio, MWClassID.DOUBLE);
		MWNumericArray lambdaIn = new MWNumericArray(lambda, MWClassID.DOUBLE);
		MWNumericArray seedIn = new MWNumericArray(seed, MWClassID.DOUBLE);

		// When returning values, the first parameter of _run method must be the number
		// of returned values
		int nValuesReturned = 2;

		// Call in Matlab:
		// [Centroids, Weights] = MIMLNN_run_train(train_bags,train_target,ratio,
		// lambda, seed);

		/**
		 * The trained classifier being:
		 * <ul>
		 * <li>classifier[0] the centroids. A Kx1 cell structure, where the k-th
		 * centroid of the RBF neural network is stored in Centroid{k,1}</li>
		 * <li>classifier[1] the weights. A (K+1)xQ matrix used for label
		 * prediction</li>
		 * </ul>
		 */
		classifier = mimlnn.MIMLNN_run_train(nValuesReturned, train_bags, train_targets, ratioIn, lambdaIn, seedIn);

		// Dispose of native MW resources
		ratioIn.dispose();
		lambdaIn.dispose();
		seedIn.dispose();

	}

	@Override
	protected Object[] predictMWClassifier(MWCellArray train_bags, MWNumericArray train_targets,
			MWNumericArray test_bag) throws MWException {

		// When returning values, the first parameter of _run method must be the number
		// of returned values
		int nValuesReturned = 2;

		// Call in Matlab:
		// [Outputs,Pre_Labels] = MIMLNN_run_test(aBag, Centroids, Weights);
		Object[] prediction = mimlnn.MIMLNN_run_test(nValuesReturned, test_bag, classifier[0], classifier[1]);

		return prediction;
	}
}
