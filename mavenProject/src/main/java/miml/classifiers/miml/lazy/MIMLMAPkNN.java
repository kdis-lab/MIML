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

import mulan.classifier.lazy.MLkNN;

/**
 * MIMLMAPkNN is the adaptation to the MIML framework of the MLkNN[1]
 * multi-label algorithm. To perform this adaptation, MIMLMAPkNN maintains the
 * treatment of labels of MLkNN but uses a multi-instance measure of distance.
 * 
 * <em> [1] Min-Ling Zhang, Zhi-Hua Zhou (2007). ML-KNN: A lazy learning
 * approach to multi-label learning. Pattern Recogn.. 40(7):2038--2048. </em>
 */
public class MIMLMAPkNN extends MultiInstanceMultiLabelKNN {

	/**
	 * Generated Serial version UID.
	 */
	private static final long serialVersionUID = 1L;

	/** Smooth factor */
	protected double smooth = 1.0;

	/**
	 * Default constructor.
	 * 
	 * @param metric The distance metric between bags considered by the classifier.
	 */
	public MIMLMAPkNN(MIMLDistanceFunction metric) {
		super(metric, 10);
		this.smooth = 1.0;
		this.classifier = new MLkNN(10, smooth);
	}

	/**
	 * A constructor that sets the number of neighbours.
	 *
	 * @param metric          The distance metric between bags considered by the
	 *                        classifier.
	 * @param numOfNeighbours The number of neighbours.
	 */
	public MIMLMAPkNN(int numOfNeighbours, MIMLDistanceFunction metric) {
		super(metric, numOfNeighbours);
		this.smooth = 1.0;
		this.classifier = new MLkNN(numOfNeighbours, smooth);
	}

	/**
	 * A constructor that sets the number of neighbours and the value of smooth.
	 *
	 * @param metric          The distance metric between bags considered by the
	 *                        classifier.
	 * @param numOfNeighbours The number of neighbours.
	 * @param smooth          The smooth factor.
	 */
	public MIMLMAPkNN(int numOfNeighbours, double smooth, MIMLDistanceFunction metric) {
		super(metric, numOfNeighbours);
		this.smooth = smooth;
		this.classifier = new MLkNN(numOfNeighbours, smooth);
	}

	/**
	 * No-arg constructor for xml configuration
	 */
	public MIMLMAPkNN() {
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
		this.smooth = configuration.getDouble("smooth", 1.0);
		this.classifier = new MLkNN(numOfNeighbours, smooth);
	}

	/**
	 * Gets the smooth factor considered by the classifier.
	 *
	 * @return the smooth factor
	 */
	public double getSmooth() {
		return smooth;
	}

	/**
	 * Sets the smooth factor considered by the classifier.
	 *
	 * @param smooth the new smooth factor
	 */
	public void setSmooth(double smooth) {
		this.smooth = smooth;
	}
}
