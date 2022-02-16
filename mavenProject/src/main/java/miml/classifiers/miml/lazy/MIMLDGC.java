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

import miml.classifiers.ml.MLDGC;

/**
 * MIMLDGC is the adaptation to the MIML framework of the MLDGC[1] multi-label
 * algorithm. To perform this adaptation, MIMLDGC maintains the treatment of
 * labels of MLDGC but computes the proximity between bags with a multi-instance
 * measure of distance.
 * 
 * <em> [1] Oscar Reyes, Carlos Morell, Sebastián Ventura (2016). Effective lazy
 * learning algorithm based on a data gravitation model for multi-label
 * learning. Information Sciences. Vol 340, issue C. </em>
 */
public class MIMLDGC extends MultiInstanceMultiLabelKNN {

	/** For serialization. */
	private static final long serialVersionUID = 1L;

	/**
	 * No-arg constructor for xml configuration
	 */
	public MIMLDGC() {
	}

	/**
	 * Default constructor.
	 * 
	 * @param metric The distance metric between bags considered by the classifier.
	 */

	public MIMLDGC(MIMLDistanceFunction metric) {
		super(metric, 10);
		this.classifier = new MLDGC(10);
	}

	/**
	 * A constructor that sets the number of neighbours.
	 *
	 * @param metric          The distance metric between bags considered by the
	 *                        classifier.
	 * @param numOfNeighbours the number of neighbours.
	 */
	public MIMLDGC(MIMLDistanceFunction metric, int numOfNeighbours) {
		super(metric, numOfNeighbours);
		this.classifier = new MLDGC(numOfNeighbours);
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
		boolean extNeigh = configuration.getBoolean("extendedNeighborhood", false);
		this.classifier = new MLDGC(numOfNeighbours);
		((MLDGC) classifier).setExtNeigh(extNeigh);
	}

}
