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
package miml.classifiers.miml.optimization;

import org.apache.commons.configuration2.Configuration;

import com.mathworks.toolbox.javabuilder.MWCellArray;
import com.mathworks.toolbox.javabuilder.MWClassID;
import com.mathworks.toolbox.javabuilder.MWException;
import com.mathworks.toolbox.javabuilder.MWNumericArray;

import MWAlgorithms.MWMIMLWel;
import miml.classifiers.miml.MWClassifier;

/**
 * Wrapper for Matlab
 * <a href="http://www.lamda.nju.edu.cn/code_MIMLWEL.ashx">MIMLFast</a>
 * algorithm for MIML data.<br>
 * See: <em>S.-J. Yang, Y. Jiang, and Z.-H. Zhou. Multi-instance multi-label
 * learning with weak label.In: Proceedings of the 23rd International Joint
 * Conference on Artificial Intelligence (IJCAI'13), Beijing, China, 2013. </em>
 * 
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20220107
 */
public class MIMLWel extends MWClassifier {

	/** For serialization. */
	private static final long serialVersionUID = 7348850142116317625L;

	/** A matlab object wrapping the MIMLWel algorithm. */
	MWMIMLWel mimlwel;

	/** Controls the empirical loss on labeled data. */
	double opts_C = 50;

	/**
	 * Controls the difference between the learned training targets and the original
	 * input training targets.
	 */
	double opts_m = 1;

	/** Controls the similarity between training_bags and their prototypes. */
	double opts_beta = 2;

	/** Iteration number. */
	double opts_iteration = 20;

	/** Value for epsilon. */
	double opts_epsilon = 1e-3;

	/**
	 * The number of centroids of the i-th class is set to be ratio*Ti, where Ti is
	 * the number of train bags with label i.
	 */
	double ratio = 0.1;

	/**
	 * The ratio used to determine the standard deviation of the Gaussian activation
	 * function.
	 */
	double mu = 1.0;

	/**
	 * No-argument constructor for xml configuration.
	 * 
	 * @throws MWException To be handled.
	 */
	public MIMLWel() throws MWException {
		super();
		mimlwel = new MWMIMLWel();
	}

	/**
	 * Constructor initializing fields of MIMLWel.
	 * 
	 * @param opts_C         Value for the opts_C field.
	 * @param opts_m         Value for the opts_m field.
	 * @param opts_beta      Value for the opts_beta field.
	 * @param opts_iteration Value for the opts_iteration field.
	 * @param opts_epsilon   Value for the opts_epsilon field.
	 * @param ratio          Value for the ratio field.
	 * @param mu             Value for the mu field.
	 * @throws MWException To be handled in upper level.
	 */

	public MIMLWel(double opts_C, double opts_m, double opts_beta, double opts_iteration, double opts_epsilon,
			double ratio, double mu) throws MWException {
		this();
		this.opts_C = opts_C;
		this.opts_m = opts_m;
		this.opts_beta = opts_beta;
		this.opts_iteration = opts_iteration;
		this.opts_epsilon = opts_epsilon;
		this.ratio = ratio;
		this.mu = mu;
	}

	@Override
	public void dispose() {
		// Dispose of native MW resources
		mimlwel.dispose();
	}

	@Override
	public void configure(Configuration configuration) {
		this.opts_C = configuration.getDouble("C", 50);
		this.opts_m = configuration.getDouble("m", 1);
		this.opts_beta = configuration.getDouble("beta", 2);
		this.opts_iteration = configuration.getDouble("iteration", 20);
		this.opts_epsilon = configuration.getDouble("epsilon", 1e-3);
		this.ratio = configuration.getDouble("ratio", 0.1);
		this.mu = configuration.getDouble("mu", 1.0);
	}

	@Override
	protected void trainMWClassifier(MWCellArray train_bags, MWNumericArray train_targets) throws MWException {

		MWNumericArray opts_CIn = new MWNumericArray(opts_C, MWClassID.DOUBLE);
		MWNumericArray opts_mIn = new MWNumericArray(opts_m, MWClassID.DOUBLE);
		MWNumericArray opts_betaIn = new MWNumericArray(opts_beta, MWClassID.DOUBLE);
		MWNumericArray opts_iterationIn = new MWNumericArray(opts_iteration, MWClassID.DOUBLE);
		MWNumericArray opts_epsilonIn = new MWNumericArray(opts_epsilon, MWClassID.DOUBLE);
		MWNumericArray ratioIn = new MWNumericArray(ratio, MWClassID.DOUBLE);
		MWNumericArray muIn = new MWNumericArray(mu, MWClassID.DOUBLE);

		// When returning values, the first parameter of _run method must be the number
		// of returned values
		int nValuesReturned = 5;

		// Call in Matlab:
		// [Centroids,centroid_index,num_cluster,Dist,model]=MIMLWel_run_train(train_bags,train_targets,opts_C,
		// opts_m, opts_beta,
		// opts_iteration, opts_epsilon, ratio, mu);
		classifier = mimlwel.MIMLWel_run_train(nValuesReturned, train_bags, train_targets, opts_CIn, opts_mIn,
				opts_betaIn, opts_iterationIn, opts_epsilonIn, ratioIn, muIn);

		// Dispose of native MW resources
		opts_CIn.dispose();
		opts_mIn.dispose();
		opts_betaIn.dispose();
		opts_iterationIn.dispose();
		opts_epsilonIn.dispose();
		ratioIn.dispose();
		muIn.dispose();

	}

	@Override
	protected Object[] predictMWClassifier(MWCellArray train_bags, MWNumericArray train_targets,
			MWNumericArray test_bag) throws MWException {

		MWNumericArray muIn = new MWNumericArray(mu, MWClassID.DOUBLE);

		// When returning values, the first parameter of _run method must be the number
		// of returned values
		int nValuesReturned = 2;

		// Call in Matlab:
		// [Outputs,Pre_Labels]=MIMLWel_run_test(aBag, mu,
		// Centroids,centroid_index,num_cluster,Dist, model)
		Object[] prediction = mimlwel.MIMLWel_run_test(nValuesReturned, test_bag, muIn, classifier[0], classifier[1],
				classifier[2], classifier[3], classifier[4]);

		// Dispose of native MW resources

		return prediction;
	}

	/**
	 * Gets the value of the opts_C property.
	 * 
	 * @return double
	 */
	public double getOpts_C() {
		return opts_C;
	}

	/**
	 * Sets the value of the opts_C property.
	 * 
	 * @param opts_C The new value for the property.
	 */
	public void setOpts_C(int opts_C) {
		this.opts_C = opts_C;
	}

	/**
	 * Gets the value of the opts_m property.
	 * 
	 * @return double
	 */
	public double getOpts_m() {
		return opts_m;
	}

	/**
	 * Sets the value of the opts_m property.
	 * 
	 * @param opts_m The new value for the property.
	 */
	public void setOpts_m(double opts_m) {
		this.opts_m = opts_m;
	}

	/**
	 * Gets the value of the opts_beta property.
	 * 
	 * @return double
	 */
	public double getOpts_beta() {
		return opts_beta;
	}

	/**
	 * Sets the value of the beta property.
	 * 
	 * @param opts_beta The new value for the property.
	 */
	public void setOpts_beta(double opts_beta) {
		this.opts_beta = opts_beta;
	}

	/**
	 * Gets the value of the opts_iteration property.
	 * 
	 * @return double
	 */
	public double getOpts_iteration() {
		return opts_iteration;
	}

	/**
	 * Sets the value of the opts_iteration property.
	 * 
	 * @param opts_iteration The new value for the property.
	 */
	public void setOpts_iteration(int opts_iteration) {
		this.opts_iteration = opts_iteration;
	}

	/**
	 * Gets the value of the opts_epsilon property.
	 * 
	 * @return double
	 */
	public double getOpts_epsilon() {
		return opts_epsilon;
	}

	/**
	 * Sets the value of the opts_epsilon property.
	 * 
	 * @param opts_epsilon The new value for the property.
	 */
	public void setOpts_epsilon(double opts_epsilon) {
		this.opts_epsilon = opts_epsilon;
	}

	/**
	 * Gets the value of the ratio property.
	 * 
	 * @return double
	 */
	public double getRatio() {
		return ratio;
	}

	/**
	 * Sets the value of the ratio property.
	 * 
	 * @param ratio The new value for the property.
	 */
	public void setRatio(double ratio) {
		this.ratio = ratio;
	}

	/**
	 * Gets the value of the mu property.
	 * 
	 * @return double
	 */
	public double getMu() {
		return mu;
	}

	/**
	 * Sets the value of the mu property.
	 * 
	 * @param mu The new value for the property.
	 */
	public void setMu(double mu) {
		this.mu = mu;
	}

}
