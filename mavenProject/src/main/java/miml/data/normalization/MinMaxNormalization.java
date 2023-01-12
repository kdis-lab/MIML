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
package miml.data.normalization;


import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import weka.core.Instance;

/**
 * 
 * Class implementing min-max normalization for MIML datasets.
 * 
 * @author Eva Gibaja
 * @version 20220613
 *
 */

public class MinMaxNormalization {

	/** Max, Min and Range values for features. */
	protected double Max[] = null;
	protected double Min[] = null;
	protected double Range[] = null;

	/** Number of features of the bags in the MIML dataset. */
	int nFeatures = -1;

	/**
	 * Value indicating if the bag attributes of the dataset were normalized before
	 * calling normalize (e.g. the dataset does not need normalization).
	 */
	boolean normalized = true;

	/**
	 * Applies min-max normalization on a MIMLInstances dataset. Given an attribute
	 * values, x, the new x' value will be x' = (x-min(x))/(max(x)-min(x)). Before
	 * call this method the method update stats must be called to get the max and
	 * min values for attributes.
	 * 
	 * @param mimlDataSet a dataset to normalize.
	 * @throws Exception To be handled in upper level.
	 */
	public void normalize(MIMLInstances mimlDataSet) throws Exception {
		if (Max == null)
			throw new Exception("\nThe stats have not been updated. Call updateStats first.");
		for (int i = 0; i < mimlDataSet.getNumBags(); i++) {

			MIMLBag bag = null;

			bag = mimlDataSet.getBag(i);

			for (int j = 0; j < bag.getNumInstances(); j++) {
				Instance instance = mimlDataSet.getInstance(i, j);

				for (int k = 0; k < instance.numAttributes(); k++) {
					
					double value = instance.value(k);			
					
					// to avoid dividing by zero in case of a 0 range
					if(Double.compare(Min[k], Max[k])!=0) 
					{						
						value = (value - Min[k]) / (Range[k]);

					} else {
						
						value = 1;						
					}
					instance.setValue(k, value);
				}
			}
		}
	}

	/**
	 * Set the max and min values for all attributes in the bag. This method must be
	 * called before call normalized. If several datasets with the same structure
	 * are normalized at once (e.g. train and test or folds partitioned files), this
	 * method can be called for each dataset before normalization. Besides, if the
	 * method method detects that all the attributes are jet normalized, it sets the
	 * "normalized" property as true.
	 * 
	 * <code>
	 *  MinMaxNormalization norm = new MinMaxNormalization();
	 *  MIMLInstances mimlDataSet1 = new MIMLInstances("toy_train.arff", "toy.xml");
	 *  MIMLInstances mimlDataSet2 = new MIMLInstances("toy_test.arff", "toy.xml");
	 *  norm.updateStats(mimlDataSet1); norm.updateStats(mimlDataSet2);
	 *  if (norm.isNormalized() == false)
	 *  { 
	 *      norm.normalize(mimlDataSet1);
	 *      norm.normalize(mimlDataSet2);
	 *  }
	 *  </code>
	 * 
	 * @param mimlDataSet MIML dataset.
	 * @throws Exception To be handled in upper level.
	 */
	public void updateStats(MIMLInstances mimlDataSet) throws Exception {

		if (Max == null) {
			nFeatures = mimlDataSet.getBag(1).getNumAttributesInABag();

			Max = new double[nFeatures];
			Min = new double[nFeatures];
			Range = new double[nFeatures];

			for (int i = 0; i < nFeatures; i++) {
				Max[i] = Double.NEGATIVE_INFINITY;
				Min[i] = Double.POSITIVE_INFINITY;
				Range[i] = 0;
			}
		}

		for (int i = 0; i < mimlDataSet.getNumBags(); i++) {

			MIMLBag bag = null;

			bag = mimlDataSet.getBag(i);

			for (int j = 0; j < bag.getNumInstances(); j++) {

				Instance instance = mimlDataSet.getInstance(i, j);

				for (int k = 0; k < instance.numAttributes(); k++) {

					if (instance.attribute(k).isNumeric()) {
						if (instance.value(k) < 0 || instance.value(k) > 1)
							normalized = false;
						if (instance.value(k) < Min[k])
							Min[k] = instance.value(k);
						if (instance.value(k) > Max[k])
							Max[k] = instance.value(k);
					}
				}
			}
		}

		for (int i = 0; i < nFeatures; i++) {
			Range[i] = Max[i] - Min[i];
		}
	}

	/**
	 * Returns true if the dataset does not need normalization. Requires a previous
	 * call of updateStats.
	 * @return boolean
	 */
	public boolean isNormalized() {
		return normalized;
	}


	/**
	 * Retuns an array with the maximum values for all bag attributes in the
	 * dataset. Requires a previous call of updateStats.
	 * @return double[]
	 */
	public double[] getMax() {
		return Max;
	}

	/**
	 * Retuns an array with the minimum values for all bag attributes in the
	 * dataset. Requires a previous call of updateStats.
	 * @return double[]
	 */
	public double[] getMin() {
		return Min;
	}

	/**
	 * Retuns an array with the range values (i.e. max-min) for all bag attributes
	 * in the dataset. Requires a previous call of updateStats.
	 * @return double
	 */
	public double[] getRange() {
		return Range;
	}

	/**
	 * Retuns the number of bag attributes in the dataset. Requires a previous call
	 * of updateStats.
	 * @return int
	 */
	public int getnFeatures() {
		return nFeatures;
	}
}
