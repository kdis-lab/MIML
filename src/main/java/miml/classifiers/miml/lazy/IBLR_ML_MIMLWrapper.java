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
package miml.classifiers.miml.lazy;

import org.apache.commons.configuration2.Configuration;

import mulan.classifier.lazy.IBLR_ML;


/**
 * Wrapper for IBLR-ML and IBLR-ML+ methods of Mulan Library. For more
 * information, see<br>
 * <br>
 * Weiwei Cheng, Eyke Hullermeier (2009). Combining instance-based learning and
 * logistic regression for multilabel classification. Machine Learning.
 * 76(2-3):211-225. <br>
 */

public class IBLR_ML_MIMLWrapper extends MultiLabelKNN_MIMLWrapper {
	/**
	 * Generated Serial version UID.
	 */
	private static final long serialVersionUID = -7626000650278873479L;

	/**
	 * By default, IBLR-ML is used (addFeatures is false). One can change to IBLR-ML+
	 * through the constructor.
	 */
	private boolean addFeatures = false;

	/**
	 * Default constructor.
	 * 
	 * @param metric
	 *            The distance metric between bags considered by the classifier.
	 */
	public IBLR_ML_MIMLWrapper(DistanceFunction_MIMLWrapper metric) {
		super(metric, 10);
		this.addFeatures = false;
		this.classifier = new IBLR_ML(numOfNeighbors, addFeatures);
	}

	/**
	 * A constructor that sets the number of neighbors.
	 *
	 * @param metric
	 *            The distance metric between bags considered by the classifier.
	 * @param numOfNeighbors
	 *            The number of neighbors.
	 */
	public IBLR_ML_MIMLWrapper(int numOfNeighbors, DistanceFunction_MIMLWrapper metric) {
		super(metric, numOfNeighbors);
		this.addFeatures = false;
		this.classifier = new IBLR_ML(numOfNeighbors, addFeatures);
	}

	/**
	 * A constructor that sets the number of neighbors and whether IBLR-ML or
	 * IBLR-ML+ is used.
	 *
	 * @param metric
	 *            The distance metric between bags considered by the classifier.
	 * @param numOfNeighbors
	 *            The number of neighbors.
	 * @param addFeatures
	 *            If false IBLR-ML is used. If true, IBLR-ML+ is used.
	 */
	public IBLR_ML_MIMLWrapper(int numOfNeighbors, boolean addFeatures, DistanceFunction_MIMLWrapper metric) {
		super(metric, numOfNeighbors);
		this.addFeatures = addFeatures;
		this.classifier = new IBLR_ML(numOfNeighbors, addFeatures);
	}

	/**
	 * No-arg constructor for xml configuration
	 */
	public IBLR_ML_MIMLWrapper() {
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see miml.core.IConfiguration#configure(org.apache.commons.configuration2.
	 * Configuration)
	 */
	@Override
	public void configure(Configuration configuration) {

		super.configure(configuration);
		this.addFeatures = configuration.getBoolean("addFeatures");
		this.classifier = new IBLR_ML(numOfNeighbors, addFeatures);
	}

	/**
	 * Gets the value of addFeatures.  If false IBLR-ML is used. If true, IBLR-ML+ is used.
	 *
	 * @return The value of addFeatures.
	 */
	public boolean getAddFeatures() {
		return addFeatures;
	}

	/**
	 * Sets the value of AddFeatures. If false IBLR-ML is used. If true, IBLR-ML+ is used.
	 *
	 * @param addFeatures
	 *            The new value of addFeatures.
	 */
	public void setAddFeatures(boolean addFeatures) {
		this.addFeatures = addFeatures;
	}

}
