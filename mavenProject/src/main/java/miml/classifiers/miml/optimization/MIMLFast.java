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

import MWAlgorithms.MWMIMLFast;
import miml.classifiers.miml.MWClassifier;

/**
 * Wrapper for Matlab
 * <a href="http://www.lamda.nju.edu.cn/code_MIMLfast.ashx">MIMLFast</a>
 * algorithm for MIML data.<br>
 * See: <em>S.-J. Huang W. Gao and Z.-H. Zhou. Fast multi-instance multi-label
 * learning. In: Proceedings of the 28th AAAI Conference on Artificial
 * Intelligence (AAAI'14), 2014. </em>
 * 
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20220107
 */
public class MIMLFast extends MWClassifier {

	/** For serialization. */
	private static final long serialVersionUID = 988010931573561210L;

	/** A matlab object wrapping the MIMLFast algorithm. */
	static MWMIMLFast mimlfast;

	/** Dimension of the shared space. */
	int D = 100;

	/** Norm of each vector. */
	int norm_up = 10;

	/** Number of iterations. */
	int maxiter = 10;

	/** Step size of SGD (stochastic gradient descent). */
	double step_size = 0.005;

	/** Lambda. */
	double lambda = 1e-5;

	/** Number of sub concepts. */
	int num_sub = 5;

	/***/
	int opts_norm = 1;

	/***/
	int opts_average_size = 10;

	/***/
	int opts_average_begin = 0;

	/**
	 * No-argument constructor for xml configuration.
	 * 
	 * @throws MWException To be handled.
	 */
	public MIMLFast() throws MWException {
		super();
		mimlfast = new MWMIMLFast();
	}

	/**
	 * Constructor setting several properties.
	 * 
	 * @param d         Value for d.
	 * @param norm_up   Value for norm_up.
	 * @param maxiter   Value for maxiter.
	 * @param step_size Value for step_size.
	 * @param num_sub   Value for num_sub.
	 * @throws MWException To be handled in upper level.
	 */
	public MIMLFast(int d, int norm_up, int maxiter, double step_size, int num_sub) throws MWException {
		this();
		D = d;
		this.norm_up = norm_up;
		this.maxiter = maxiter;
		this.step_size = step_size;
		this.num_sub = num_sub;
	}

	/**
	 * Constructor setting several properties.
	 * 
	 * @param d                  Value for d.
	 * @param norm_up            Value for norm_up.
	 * @param maxiter            Value for maxiter.
	 * @param step_size          Value for step_size.
	 * @param num_sub            Value for num_sub.
	 * @param lambda             Value for lambda.
	 * @param opts_norm          Value for opts_norm.
	 * @param opts_average_size  Value for opts_average_size.
	 * @param opts_average_begin Value for opts_average_begin.
	 * @throws MWException To be handled in upper level.
	 */
	public MIMLFast(int d, int norm_up, int maxiter, double step_size, double lambda, int num_sub, int opts_norm,
			int opts_average_size, int opts_average_begin) throws MWException {
		this(d, norm_up, maxiter, step_size, num_sub);
		this.lambda = lambda;
		this.opts_norm = opts_norm;
		this.opts_average_size = opts_average_size;
		this.opts_average_begin = opts_average_begin;
	}

	@Override
	public void dispose() {
		// Dispose of native MW resources
		mimlfast.dispose();
	}

	@Override
	public void configure(Configuration configuration) {
		this.D = configuration.getInt("D", 100);
		this.norm_up = configuration.getInt("normUp", 10);
		this.maxiter = configuration.getInt("maxiter", 10);
		this.step_size = configuration.getDouble("stepSize", 0.005);
		this.num_sub = configuration.getInt("numSub", 5);
		this.lambda = configuration.getDouble("lambda", 1e-5);
		this.opts_norm = configuration.getInt("norm", 1);
		this.opts_average_size = configuration.getInt("averageSize", 10);
		this.opts_average_begin = configuration.getInt("averageBegin", 0);
	}

	@Override
	protected void trainMWClassifier(MWCellArray train_bags, MWNumericArray train_targets) throws MWException {

		MWNumericArray DIn = new MWNumericArray(D, MWClassID.DOUBLE);
		MWNumericArray norm_upIn = new MWNumericArray(norm_up, MWClassID.DOUBLE);
		MWNumericArray maxiterIn = new MWNumericArray(maxiter, MWClassID.DOUBLE);
		MWNumericArray step_sizeIn = new MWNumericArray(step_size, MWClassID.DOUBLE);
		MWNumericArray lambdaIn = new MWNumericArray(lambda, MWClassID.DOUBLE);
		MWNumericArray num_subIn = new MWNumericArray(num_sub, MWClassID.DOUBLE);
		MWNumericArray opts_normIn = new MWNumericArray(opts_norm, MWClassID.DOUBLE);
		MWNumericArray opts_average_sizeIn = new MWNumericArray(opts_average_size, MWClassID.DOUBLE);
		MWNumericArray opts_average_beginIn = new MWNumericArray(opts_average_begin, MWClassID.DOUBLE);

		// When returning values, the first parameter of _run method must be the number
		// of returned values
		int nValuesReturned = 3;

		// Call in Matlab:
		// [AW,AV,Anum]=MIMLFast_run_train(train_bags,train_targets,D, norm_up, maxiter,
		// step_size, lambda, num_sub, opts_norm, opts_average_size,
		// opts_average_begin);

		/**
		 * The trained classifier being:
		 * <ul>
		 * <li>classifier[0] AW. A numBagsxnumFeaturesperBag array of double.</li>
		 * <li>classifier[1] AV. An array of double.</li>
		 * <li>classifier[2] Anum. An integer value.</li>
		 * </ul>
		 */
		classifier = mimlfast.MIMLFast_run_train(nValuesReturned, train_bags, train_targets, DIn, norm_upIn, maxiterIn,
				step_sizeIn, lambdaIn, num_subIn, opts_normIn, opts_average_sizeIn, opts_average_beginIn);

		// Dispose of native MW resources
		DIn.dispose();
		norm_upIn.dispose();
		maxiterIn.dispose();
		step_sizeIn.dispose();
		lambdaIn.dispose();
		num_subIn.dispose();
		opts_normIn.dispose();
		opts_average_sizeIn.dispose();
		opts_average_beginIn.dispose();
	}

	@Override
	protected Object[] predictMWClassifier(MWCellArray train_bags, MWNumericArray train_targets,
			MWNumericArray test_bag) throws MWException {

		MWNumericArray num_subIn = new MWNumericArray(num_sub, MWClassID.DOUBLE);

		// When returning values, the first parameter of _run method must be the number
		// of returned values
		int nValuesReturned = 2;

		// Call in Matlab:
		// [Outputs,Pre_Labels]=MIMLFast_run_test(aBag, AW,AV,Anum, num_sub)
		Object[] prediction = mimlfast.MIMLFast_run_test(nValuesReturned, test_bag, classifier[0], classifier[1],
				classifier[2], num_subIn);

		// Dispose of native MW resources
		num_subIn.dispose();

		return prediction;
	}

	/**
	 * Gets the value of the D property.
	 * 
	 * @return int
	 */
	public int getD() {
		return D;
	}

	/**
	 * Sets the value of the D property.
	 * 
	 * @param d The new value for the property.
	 */
	public void setD(int d) {
		D = d;
	}

	/**
	 * Gets the value of the norm_up property.
	 * 
	 * @return int
	 */
	public int getNorm_up() {
		return norm_up;
	}

	/**
	 * Sets the value of the norm_up property.
	 * 
	 * @param norm_up The new value for the property.
	 */
	public void setNorm_up(int norm_up) {
		this.norm_up = norm_up;
	}

	/**
	 * Gets the value of the maxiter property.
	 * 
	 * @return int
	 */
	public int getMaxiter() {
		return maxiter;
	}

	/**
	 * Sets the value of the maxiter property.
	 * 
	 * @param maxiter The new value for the property.
	 */
	public void setMaxiter(int maxiter) {
		this.maxiter = maxiter;
	}

	/**
	 * Gets the value of the step_size property.
	 * 
	 * @return double
	 */
	public double getStep_size() {
		return step_size;
	}

	/**
	 * Sets the value of the step_size property.
	 * 
	 * @param step_size The new value for the property.
	 */
	public void setStep_size(double step_size) {
		this.step_size = step_size;
	}

	/**
	 * Gets the value of the lambda property.
	 * 
	 * @return double
	 */
	public double getLambda() {
		return lambda;
	}

	/**
	 * Sets the value of the lambda property.
	 * 
	 * @param lambda The new value for the property.
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * Gets the value of the num_sub property.
	 * 
	 * @return int
	 */
	public int getNum_sub() {
		return num_sub;
	}

	/**
	 * Sets the value of the num_sub property.
	 * 
	 * @param num_sub The new value for the property.
	 */
	public void setNum_sub(int num_sub) {
		this.num_sub = num_sub;
	}

	/**
	 * Gets the value of the opts_norm property.
	 * 
	 * @return int
	 */
	public int getOpts_norm() {
		return opts_norm;
	}

	/**
	 * Sets the value of the opts_norm property.
	 * 
	 * @param opts_norm The new value for the property.
	 */
	public void setOpts_norm(int opts_norm) {
		this.opts_norm = opts_norm;
	}

	/**
	 * Gets the value of the opts_average_size property.
	 * 
	 * @return int
	 */
	public int getOpts_average_size() {
		return opts_average_size;
	}

	/**
	 * Sets the value of the opts_average_size property.
	 * 
	 * @param opts_average_size The new value for the property.
	 */
	public void setOpts_average_size(int opts_average_size) {
		this.opts_average_size = opts_average_size;
	}

	/**
	 * Gets the value of the opts_average_begin property.
	 * 
	 * @return int
	 */
	public int getOpts_average_begin() {
		return opts_average_begin;
	}

	/**
	 * Sets the value of the opts_average_begin property.
	 * 
	 * @param opts_average_begin The new value for the property.
	 */
	public void setOpts_average_begin(int opts_average_begin) {
		this.opts_average_begin = opts_average_begin;
	}

}
