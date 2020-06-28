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

/**Wrapper for ML-kNN (Multi-Label k Nearest Neighbours) algorithm of
 * Mulan Library. For more information, see: <br>
 * <br>
 * Min-Ling Zhang, Zhi-Hua Zhou (2007). ML-KNN: A lazy learning approach to multi-label learning. Pattern Recogn.. 40(7):2038--2048.
 * <br>
 * */
public class MLkNN_MIMLWrapper extends MultiLabelKNN_MIMLWrapper {

	/**
	 * Generated Serial version UID.
	 */
	private static final long serialVersionUID = 1L;

	/** Smooth factor */
	protected double smooth = 1.0;
	
	/**
	 * Default constructor.
	 * 
	 * @param metric
	 *            The distance metric between bags considered by the classifier.
	 */
	public MLkNN_MIMLWrapper(DistanceFunction_MIMLWrapper metric) {
		super(metric, 10);		
		this.smooth =1.0;
		this.classifier = new MLkNN(numOfNeighbors, smooth);
	}	
	
	/**
	 * A constructor that sets the number of neighbors.
	 *
	 * @param metric
	 *            The distance metric between bags considered by the classifier.
	 * @param numOfNeighbors
	 *            The number of neighbors.
	 */
	public MLkNN_MIMLWrapper(int numOfNeighbors, DistanceFunction_MIMLWrapper metric) {
		super(metric, numOfNeighbors);		
		this.smooth = 1.0;
		this.classifier = new MLkNN(numOfNeighbors, smooth);
	}
	
	/**
	 * A constructor that sets the number of neighbors and the value of smooth.
	 *
	 * @param metric
	 *            The distance metric between bags considered by the classifier.
	 * @param numOfNeighbors
	 *            The number of neighbors.
	 * @param smooth
	 * 	          The smooth factor.           
	 */
	public MLkNN_MIMLWrapper(int numOfNeighbors, double smooth, DistanceFunction_MIMLWrapper metric) {
		super(metric, numOfNeighbors);		
		this.smooth = smooth;		
		this.classifier = new MLkNN(numOfNeighbors, smooth);
	}			
	
	/**
	 *  No-arg constructor for xml configuration   
	*/
	public MLkNN_MIMLWrapper() {	
	}
		
	
	/*
	 * (non-Javadoc)
	 * @see miml.core.IConfiguration#configure(org.apache.commons.configuration2.Configuration)
	 */
	@Override
	public void configure(Configuration configuration) {
		
		super.configure(configuration);		
		this.smooth = configuration.getDouble("smooth");
		this.classifier = new MLkNN(numOfNeighbors, smooth);
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
