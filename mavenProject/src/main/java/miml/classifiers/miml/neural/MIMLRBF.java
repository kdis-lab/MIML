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
import MWAlgorithms.MWMIMLRBF;
import miml.classifiers.miml.MWClassifier;

/**
 * <p>
 * Class to execute the
 * <a href="http://palm.seu.edu.cn/zhangml/Resources.htm">MIMLRBF</a> algorithm
 * for MIML data. For more information, see <em>Zhang, M. L., &amp; Wang, Z. J.
 * (2009). MIMLRBF: RBF neural networks for multi-instance multi-label learning.
 * Neurocomputing, 72(16-18), 3951-3956.</em>.
 * </p>
 * 
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20211024
 */

public class MIMLRBF extends MWClassifier {

	/** For serialization. */
	private static final long serialVersionUID = -8995828753405862133L;

	/** A matlab object wrapping the mimlrbf algorithm. */
	static MWMIMLRBF mimlrbf;

	// Default parameters: ratio=0.1, mu=0.6 seed=1
	/**
	 * The number of centroids of the i-th label is set to be ratio*Ti, where Ti is
	 * the number of train bags with label i.
	 */
	double ratio = 0.1;

	/**
	 * The ratio used to determine the standard deviation of the Gaussian activation
	 * function.
	 */
	double mu = 0.6;

	/** Seed for kmedoids clustering. */
	int seed = 1;

	/**
	 * No-argument constructor for xml configuration.
	 * 
	 * @throws MWException To be handled.
	 */
	public MIMLRBF() throws MWException {
		super();
		mimlrbf = new MWMIMLRBF();
	}

	/**
	 * Basic constructor to initialize the classifier.
	 * 
	 * @param ratio The fraction parameter of MIMLRBF.
	 * @param mu    The scaling factor of MIMLRBF.
	 * @throws MWException To be handled.
	 */
	public MIMLRBF(double ratio, double mu) throws MWException {
		this();
		this.ratio = ratio;
		this.mu = mu;
	}

	/**
	 * Constructor to initialize the classifier.
	 * 
	 * @param ratio The fraction parameter of MIMLRBF.
	 * @param mu    The scaling factor of MIMLRBF.
	 * @param seed  Seed for kmedoids clustering.
	 * @throws MWException To be handled.
	 */
	public MIMLRBF(double ratio, double mu, int seed) throws MWException {
		this(ratio, mu);
		this.seed = seed;
	}

	@Override
	public void dispose() {
		// Dispose of native MW resources
		mimlrbf.dispose();
	}

	/**
	 * Returns the fraction parameter considered to build the classifier.
	 * 
	 * @return The fraction parameter considered to build the classifier.
	 */
	public double getRatio() {
		return ratio;
	}

	/**
	 * Sets the fraction parameter to build the classifier.
	 * 
	 * @param ratio The fraction parameter of MIMLRBF.
	 */
	public void setRatio(double ratio) {
		this.ratio = ratio;
	}

	/**
	 * Returns the scaling factor parameter considered to build the classifier.
	 * 
	 * @return The scaling factor parameter considered to build the classifier.
	 */
	public double getMu() {
		return mu;
	}

	/**
	 * Sets the scaling factor parameter to build the classifier.
	 * 
	 * @param mu The scaling factor of MIMLRBF.
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
	 * Returns the seed for kmedoids clustering considered to build the classifier.
	 * 
	 * @param seed Seed for kmedoids clustering.
	 */
	public void setSeed(int seed) {
		this.seed = seed;
	}

	@Override
	public void configure(Configuration configuration) {
		this.ratio = configuration.getDouble("ratio", 0.1);
		this.mu = configuration.getDouble("mu", 0.6);
		this.seed = configuration.getInt("seed", 1);
	}

	@Override
	protected void trainMWClassifier(MWCellArray train_bags, MWNumericArray train_targets) throws MWException {

		MWNumericArray ratioIn = new MWNumericArray(ratio, MWClassID.DOUBLE);
		MWNumericArray muIn = new MWNumericArray(mu, MWClassID.DOUBLE);
		MWNumericArray seedIn = new MWNumericArray(seed, MWClassID.DOUBLE);

		// When returning values, the first parameter of _run method must be the number
		// of returned values
		int nValuesReturned = 3;

		// Call in Matlab:
		// [Centroids,Sigma_value,Weights]=MIMLRBF_run_train(train_bags,train_target,ratio,mu,seed)

		/**
		 * The trained classifier being:
		 * <ul>
		 * <li>classifier[0] the centroids. A Kx1 cell structure, where the k-th
		 * centroid of the RBF neural network is stored in Centroid{k,1}</li>
		 * <li>classifier[1] the sigma value. A 1xK vector, where the sigma value for
		 * the k-th centroid is stored in Sigma_value(1,k)</li>
		 * <li>classifier[2] the weights. A (K+1)xQ matrix used for label
		 * prediction</li>
		 * </ul>
		 */
		classifier = mimlrbf.MIMLRBF_run_train(nValuesReturned, train_bags, train_targets, ratioIn, muIn, seedIn);

		// Dispose of native MW resources
		ratioIn.dispose();
		muIn.dispose();
		seedIn.dispose();
	}

	@Override
	protected Object[] predictMWClassifier(MWCellArray train_bags, MWNumericArray train_targets,
			MWNumericArray test_bag) throws MWException {

		// When returning values, the first parameter of _run method must be the number
		// of returned values
		int nValuesReturned = 2;

		// Call in Matlab:
		// [Outputs,Pre_Labels]=MIMLRBF_run_test(test_bag,Centroids,Sigma_value,Weights)
		Object[] prediction = mimlrbf.MIMLRBF_run_test(nValuesReturned, test_bag, classifier[0], classifier[1],
				classifier[2]);

		return prediction;
	}

}
