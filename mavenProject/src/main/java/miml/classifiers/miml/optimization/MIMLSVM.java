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
import com.mathworks.toolbox.javabuilder.MWCharArray;
import com.mathworks.toolbox.javabuilder.MWClassID;
import com.mathworks.toolbox.javabuilder.MWException;
import com.mathworks.toolbox.javabuilder.MWNumericArray;

import MWAlgorithms.MWMIMLSVM;
import miml.classifiers.miml.MWClassifier;

/**
 * Wrapper for Matlab
 * <a href="http://www.lamda.nju.edu.cn/code_MIML.ashx">MIMLSVM</a> algorithm
 * for MIML data.<br>
 * See: <em> Z.-H. Zhou and M.-L. Zhang. Multi-instance multi-label learning
 * with application to scene classification. In: Advances in Neural Information
 * Processing Systems 19 (NIPS'06) (Vancouver, Canada) Cambridge, MA: MIT Press,
 * 2007.BIOwulf Technologies, 2001. </em> It employs Libsvm compiled for Windows
 * 64 bits (available at href="https://www.csie.ntu.edu.tw/~cjlin/libsvm/) as
 * the base learners.
 * 
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20220107
 */

public class MIMLSVM extends MWClassifier {

	/** For serialization. */
	private static final long serialVersionUID = -6743854375499783568L;

	/** A matlab object wrapping the MIMLSVM algorithm. */
	MWMIMLSVM mimlsvm;

	/**
	 * Gaussian kernel SVM. The type of svm used in training, which can take the
	 * value of "RBF", "Poly" or "Linear".
	 */
	String type = "RBF";

	/**
	 * A string that gives the corresponding parameters used for the svm:
	 * <ul>
	 * <li>If type is "RBF", para gives the value of gamma (i.e. para="1") where the
	 * kernel is exp(-Gamma*|x(i)-x(j)|^2).</li>
	 * <li>If type is "Poly", then para gives the value of gamma, coefficient, and
	 * degree respectively, where the kernel is
	 * (gamma*&lt;x(i),x(j)&gt;+coefficient)^degree. Values in the string are
	 * delimited by blank spaces (i.e. para="1, 0, 1").</li>
	 * <li>If type is "Linear", then para is an empty string, where the kernel is
	 * &lt;x(i),x(j)&gt; (i.e. para ="").</li>
	 * </ul>
	 */
	String para = "0.2";

	/** The cost parameter used for the base svm classifier. */
	double cost = 1;

	/** Whether to use the shrinking heuristics, 0 or 1 (default 1). */
	double h = 1;

	/** Parameter k is set to be 20% of the number of training bags. */
	double ratio = 0.2;

	/** Seed for kmedoids clustering. */
	double seed = 1;

	/**
	 * No-argument constructor for xml configuration.
	 * 
	 * @throws MWException To be handled.
	 */
	public MIMLSVM() throws MWException {
		super();
		mimlsvm = new MWMIMLSVM();
	}

	/**
	 * Constructor initializing fields of MIMLSVM.
	 * 
	 * @param type  Value for type field.
	 * @param para  Value for para field.
	 * @param cost  Value for cost field.
	 * @param h     Value for h field.
	 * @param ratio Value for ratio field.
	 * @param seed  Value for seed field.
	 * @throws MWException To be handled in upper level.
	 */
	public MIMLSVM(String type, String para, double cost, double h, double ratio, double seed) throws MWException {
		this();
		this.type = type;
		this.para = para;
		this.cost = cost;
		this.h = h;
		this.ratio = ratio;
		this.seed = seed;
	}

	@Override
	public void configure(Configuration configuration) {
		this.type = configuration.getString("type", "RBF");
		this.para = configuration.getString("para", "0.2");
		this.cost = configuration.getDouble("cost", 1);
		this.h = configuration.getDouble("h", 1);
		this.ratio = configuration.getDouble("ratio", 0.2);
		this.seed = configuration.getDouble("seed", 1);
	}

	@Override
	protected Object[] trainMWClassifier(MWCellArray train_bags, MWNumericArray train_targets) throws MWException {

		MWCharArray typeIn = new MWCharArray(type);
		MWCharArray paraIn = new MWCharArray(para);
		MWNumericArray costIn = new MWNumericArray(cost, MWClassID.DOUBLE);
		MWNumericArray hIn = new MWNumericArray(h, MWClassID.DOUBLE);
		MWNumericArray ratioIn = new MWNumericArray(ratio, MWClassID.DOUBLE);
		MWNumericArray seedIn = new MWNumericArray(seed, MWClassID.DOUBLE);

		// When returning values, the first parameter of _run method must be the number
		// of returned values
		int nValuesReturned = 2;

		// Call in Matlab:
		// [models, clustering]=MIMLSVM_run_train(train_bags,train_targets,type, para,
		// cost, ratio, seed, h)
		Object[] model = mimlsvm.MIMLSVM_run_train(nValuesReturned, train_bags, train_targets, typeIn, paraIn, costIn,
				ratioIn, seedIn, hIn);

		return model;
	}

	@Override
	protected Object[] predictMWClassifier(MWCellArray train_bags, MWNumericArray train_targets,
			MWNumericArray test_bag) throws MWException {

		// When returning values, the first parameter of _run method must be the number
		// of returned values
		int nValuesReturned = 2;

		// Call in Matlab:
		// [Outputs,Pre_Labels]=MIMLSVM_run_test(aBag, train_bags, clustering, models);
		Object[] prediction = mimlsvm.MIMLSVM_run_test(nValuesReturned, test_bag, train_bags, classifier[1],
				classifier[0]);

		return prediction;
	}

	/**
	 * Gets the value of the type property.
	 * 
	 * @return String
	 */
	public String getType() {
		return type;
	}

	/**
	 * Sets the value of the type property.
	 * 
	 * @param type The new value for the property.
	 */
	public void setType(String type) {
		this.type = type;
	}

	/**
	 * Gets the value of the para property.
	 * 
	 * @return String
	 */
	public String getPara() {
		return para;
	}

	/**
	 * Sets the value of the para property.
	 * 
	 * @param para The new value for the property.
	 */
	public void setPara(String para) {
		this.para = para;
	}

	/**
	 * Gets the value of the cost property.
	 * 
	 * @return double
	 */
	public double getCost() {
		return cost;
	}

	/**
	 * Sets the value of the cost property.
	 * 
	 * @param cost The new value for the property.
	 */
	public void setCost(double cost) {
		this.cost = cost;
	}

	/**
	 * Gets the value of the h property.
	 * 
	 * @return double
	 */
	public double getH() {
		return h;
	}

	/**
	 * Sets the value of the h property.
	 * 
	 * @param h The new value for the property.
	 */
	public void setH(double h) {
		this.h = h;
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
	 * Gets the value of the seed property.
	 * 
	 * @return double
	 */
	public double getSeed() {
		return seed;
	}

	/**
	 * Sets the value of the seed property.
	 * 
	 * @param seed The new value for the property.
	 */
	public void setSeed(double seed) {
		this.seed = seed;
	}

}
