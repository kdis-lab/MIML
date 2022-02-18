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
 * MIMLIBLR is the adaptation to the MIML framework of the IBLR_ML[1]
 * multi-label algorithm. To perform this adaptation, MIMLIBLR maintains the
 * treatment of labels of IBLR_ML but uses a multi-instance measure of distance.
 * 
 * <em>[1] Weiwei Cheng, Eyke Hullermeier (2009). Combining instance-based
 * learning and logistic regression for multilabel classification. Machine
 * Learning. 76(2-3):211-225. </em>
 */

public class MIMLIBLR extends MultiInstanceMultiLabelKNN {
	/**
	 * Generated Serial version UID.
	 */
	private static final long serialVersionUID = -7626000650278873479L;

	/**
	 * By default, IBLR-ML is used (addFeatures is false). One can change to
	 * IBLR-ML+ through the constructor.
	 */
	private boolean addFeatures = false;

	/**
	 * Default constructor.
	 * 
	 * @param metric The distance metric between bags considered by the classifier.
	 */
	public MIMLIBLR(MIMLDistanceFunction metric) {
		super(metric, 10);
		this.addFeatures = false;
		this.classifier = new IBLR_ML(10, addFeatures);
	}

	/**
	 * A constructor that sets the number of neighbours.
	 *
	 * @param metric          The distance metric between bags considered by the
	 *                        classifier.
	 * @param numOfNeighbours The number of neighbours.
	 */
	public MIMLIBLR(int numOfNeighbours, MIMLDistanceFunction metric) {
		super(metric, numOfNeighbours);
		this.addFeatures = false;
		this.classifier = new IBLR_ML(numOfNeighbours, addFeatures);
	}

	/**
	 * A constructor that sets the number of neighbours and whether IBLR-ML or
	 * IBLR-ML+ is used.
	 *
	 * @param metric          The distance metric between bags considered by the
	 *                        classifier.
	 * @param numOfNeighbours The number of neighbours.
	 * @param addFeatures     If false IBLR-ML is used. If true, IBLR-ML+ is used.
	 */
	public MIMLIBLR(int numOfNeighbours, boolean addFeatures, MIMLDistanceFunction metric) {
		super(metric, numOfNeighbours);
		this.addFeatures = addFeatures;
		this.classifier = new IBLR_ML(numOfNeighbours, addFeatures);
	}

	/**
	 * No-arg constructor for xml configuration
	 */
	public MIMLIBLR() {
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
		this.addFeatures = configuration.getBoolean("addFeatures", false);
		this.classifier = new IBLR_ML(numOfNeighbours, addFeatures);
	}

	/**
	 * Gets the value of addFeatures. If false IBLR-ML is used. If true, IBLR-ML+ is
	 * used.
	 *
	 * @return The value of addFeatures.
	 */
	public boolean getAddFeatures() {
		return addFeatures;
	}

	/**
	 * Sets the value of AddFeatures. If false IBLR-ML is used. If true, IBLR-ML+ is
	 * used.
	 *
	 * @param addFeatures The new value of addFeatures.
	 */
	public void setAddFeatures(boolean addFeatures) {
		this.addFeatures = addFeatures;
	}

}
