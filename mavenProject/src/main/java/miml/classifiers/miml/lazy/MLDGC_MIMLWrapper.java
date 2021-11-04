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
 * Wrapper for MLDGC (Multi-Label Data Gravitation Model) algorithm, see: <br>
 * <br>
 * Oscar Reyes, Carlos Morell, Sebastián Ventura (2016). Effective lazy learning
 * algorithm based on a data gravitation model for multi-label learning.
 * Information Sciences. Vol 340, issue C. <br>
 */
public class MLDGC_MIMLWrapper extends MultiLabelKNN_MIMLWrapper {

	/** For serialization. */
	private static final long serialVersionUID = 1L;

	/**
	 * No-arg constructor for xml configuration
	 */
	public MLDGC_MIMLWrapper() {
	}

	/**
	 * Default constructor.
	 * 
	 * @param metric The distance metric between bags considered by the classifier.
	 */

	public MLDGC_MIMLWrapper(DistanceFunction_MIMLWrapper metric) {
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
	public MLDGC_MIMLWrapper(DistanceFunction_MIMLWrapper metric, int numOfNeighbours) {
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
		this.classifier = new MLDGC(numOfNeighbours);
	}

}
