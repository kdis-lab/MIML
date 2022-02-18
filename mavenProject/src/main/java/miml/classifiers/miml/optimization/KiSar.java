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

import MWAlgorithms.MWKiSar;
import miml.classifiers.miml.MWClassifier;

/**
 * Wrapper for Matlab
 * <a href="http://www.lamda.nju.edu.cn/code_KISAR.ashx">KiSar</a> algorithm for
 * MIML data.<br>
 * For more information see:<em> Y.-F. Li, J.-H. Hu, Y. Jiang, and Z.-H. Zhou.
 * Towards discovering what patterns trigger what labels. In: Proceedings of the
 * 26th AAAI Conference on Artificial Intelligence (AAAI'12), Toronto, Canada,
 * 2012.</em> It uses LIBLINEAR, compiled for Windows 64 bits see:<br>
 * <em>R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin.
 * LIBLINEAR: A library for large linear classification. Journal of Machine
 * Learning Research 9(2008), 1871-1874.</em>
 * 
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20220107
 */
public class KiSar extends MWClassifier {

	/** For serialization. */
	private static final long serialVersionUID = 8102176719988209451L;

	/** A Matlab object wrapping the KiSar algorithm. */
	MWKiSar kisar;

	/** Parameter set for liblinear. */
	double C = 500;

	/** Maximum number of optimization iterations. */
	double iteration = 20;

	/** The epsilon parameter for the algorithm. */
	double epsilon = 1e-3;

	/** Maximum number of prototypes for k_means clustering. */
	double K = 1000;

	/**
	 * Method used to build relation matrix.<br>
	 * <ul>
	 * <li>1 =&gt; the identity matrix is returned. No cooccurrences.</li>
	 * <li>2 =&gt; all labels are related.</li>
	 * <li>3 =&gt; labels i,j coocur if their coocurrence values are greater than
	 * the mean of all values in the coocurrence matrix (including main
	 * diagonal).</li>
	 * <li>4 =&gt; labels i,j coocur if their coocurrence values are greater than
	 * the mean of the coocurrence values of all labels (excluding main
	 * diagonal).</li>
	 * <li>5 =&gt; labels i,j coocur if prob(i, j) &gt; min(prob(i), prob(j))*0.1
	 * (10 percent).</li>
	 * </ul>
	 */
	double relationMethod = 4;

	/**
	 * No-argument constructor for xml configuration.
	 * 
	 * @throws MWException To be handled.
	 */
	public KiSar() throws MWException {
		super();
		kisar = new MWKiSar();
	}

	/**
	 * Constuctor initializing fields of KiSar.
	 * 
	 * @param c              parameter for liblinear
	 * @param iteration      value for iteration
	 * @param epsilon        value for epsilon
	 * @param k              Maximum number of prototypes
	 * @param relationMethod Method used to build the relationMatrix.
	 * @throws MWException to be handled in upper level.
	 */
	public KiSar(double c, double iteration, double epsilon, double k, double relationMethod) throws MWException {
		this();
		C = c;
		this.iteration = iteration;
		this.epsilon = epsilon;
		K = k;
		this.relationMethod = relationMethod;
	}

	@Override
	public void dispose() {
		// Dispose of native MW resources
		kisar.dispose();
	}

	@Override
	public void configure(Configuration configuration) {
		this.C = configuration.getDouble("C", 500);
		this.iteration = configuration.getDouble("iteration", 20);
		this.epsilon = configuration.getDouble("epsilon", 1e-3);
		this.K = configuration.getDouble("K", 1000);
		this.relationMethod = configuration.getDouble("relationMethod", 4);
	}

	@Override
	protected void trainMWClassifier(MWCellArray train_bags, MWNumericArray train_targets) throws MWException {
		MWNumericArray CIn = new MWNumericArray(C, MWClassID.DOUBLE);
		MWNumericArray iterationIn = new MWNumericArray(iteration, MWClassID.DOUBLE);
		MWNumericArray epsilonIn = new MWNumericArray(epsilon, MWClassID.DOUBLE);
		MWNumericArray KIn = new MWNumericArray(K, MWClassID.DOUBLE);
		MWNumericArray relationMethodIn = new MWNumericArray(relationMethod, MWClassID.DOUBLE);

		// When returning values, the first parameter of _run method must be the number
		// of returned values
		int nValuesReturned = 1;

		// Call in Matlab:
		// model=KiSar_run_train(train_bags, train_targets, C, iteration, epsilon, K,
		// relationMethod)
		classifier = kisar.KiSar_run_train(nValuesReturned, train_bags, train_targets, CIn, iterationIn, epsilonIn, KIn,
				relationMethodIn);

		// Dispose of native MW resources
		CIn.dispose();
		iterationIn.dispose();
		epsilonIn.dispose();
		KIn.dispose();
		relationMethodIn.dispose();

	}

	@Override
	protected Object[] predictMWClassifier(MWCellArray train_bags, MWNumericArray train_targets,
			MWNumericArray test_bag) throws MWException {
		// When returning values, the first parameter of _run method must be the number
		// of returned values
		int nValuesReturned = 2;

		// Call in Matlab:
		// [Outputs,Pre_Labels]=KiSar_run_test(aBag, model)
		Object[] prediction = kisar.KiSar_run_test(nValuesReturned, test_bag, classifier[0]);

		return prediction;
	}

	/**
	 * Gets the value of the C property.
	 * 
	 * @return double
	 */
	public double getC() {
		return C;
	}

	/**
	 * Sets the value of the C property.
	 * 
	 * @param c The new value for the property.
	 */
	public void setC(double c) {
		C = c;
	}

	/**
	 * Gets the value of the iteration property.
	 * 
	 * @return double
	 */
	public double getIteration() {
		return iteration;
	}

	/**
	 * Sets the value of the iteration property.
	 * 
	 * @param iteration The new value for the property.
	 */
	public void setIteration(double iteration) {
		this.iteration = iteration;
	}

	/**
	 * Gets the value of the epsilon property.
	 * 
	 * @return double
	 */
	public double getEpsilon() {
		return epsilon;
	}

	/**
	 * Sets the value of the epsilon property.
	 * 
	 * @param epsilon The new value for the property.
	 */

	public void setEpsilon(double epsilon) {
		this.epsilon = epsilon;
	}

	/**
	 * Gets the value of the K property.
	 * 
	 * @return double
	 */
	public double getK() {
		return K;
	}

	/**
	 * Sets the value of the k property.
	 * 
	 * @param k The new value for the property.
	 */

	public void setK(double k) {
		K = k;
	}

	/**
	 * Gets the value of the relationMethod property.
	 * 
	 * @return double
	 */
	public double getRelationMethod() {
		return relationMethod;
	}

	/**
	 * Sets the value of the relationMethod property.
	 * 
	 * @param relationMethod The new value for the property
	 */
	public void setRelationMethod(double relationMethod) {
		this.relationMethod = relationMethod;
	}

}
