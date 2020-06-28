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

import miml.classifiers.miml.MIMLClassifier;
import miml.core.distance.IDistance;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MultiLabelKNN;
import weka.core.DistanceFunction;
import weka.core.Instance;

/**Wrapper for clas MultiLabelKNN of Mulan to work with MIML data*/
public abstract class MultiLabelKNN_MIMLWrapper extends MIMLClassifier {
	
	/**
	 * For serialization.
	 */
	private static final long serialVersionUID = 1L;

   
    /** Number of neighbors used in the k-nearest neighbor algorithm.*/
    protected int numOfNeighbors;

	
	/** Metric for measure the distance between bags.*/
    protected DistanceFunction_MIMLWrapper metric;
	
	/** Mulan MultiLabelKNN classifier. */
	protected MultiLabelKNN classifier;		
	
	
	/**
	 * Constructor to initialize the classifier. It sets the numberOfNeighbors to 10
	 * @param metric         The metric used by the algorithm to measure the
	 *                       distance between bags.
	 *                       
	 */
	public MultiLabelKNN_MIMLWrapper(DistanceFunction_MIMLWrapper metric) {
		this.metric = metric;
		this.numOfNeighbors = 10;
	}	
	
	
	/**
	 * Constructor to initialize the classifier. It sets the numOfNeighbors to 10
	 * @param metric         The metric used by the algorithm to measure the
	 *                       distance between bags.
	 * @param numOfNeighbors The number of neighbors.                       
	 */
	public MultiLabelKNN_MIMLWrapper(DistanceFunction_MIMLWrapper metric, int numOfNeighbors)
	{ 
		this.metric = metric;
		this.numOfNeighbors = numOfNeighbors;		
	}
	
	/**
	 *  No-arg constructor for xml configuration   
	*/
	public MultiLabelKNN_MIMLWrapper()
	{	
	}
	
	@SuppressWarnings("unchecked")
	@Override
	public void configure(Configuration configuration) {
		
		this.numOfNeighbors = configuration.getInt("numOfNeighbors");
		try {
			//Get the name of the metric class
			String metricName = configuration.getString("metric[@name]");
			//Instance class
			Class<? extends IDistance> metricClass =
					(Class <? extends IDistance>) Class.forName(metricName);
			
			this.metric = new DistanceFunction_MIMLWrapper(metricClass.newInstance());
		
		}
		catch(Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		
	}

	@Override
	protected void buildInternal(MIMLInstances trainingSet) throws Exception {
			if(classifier==null)		        
		            throw new Exception("The MLkNN classifier is null.");		    
			classifier.setDfunc(metric);
			
			classifier.build(trainingSet.getMLDataSet());		
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(MIMLBag instance) throws Exception, InvalidDataException {
		
		Instance bagAsInstance = instance;
		MultiLabelOutput predictions = classifier.makePrediction(bagAsInstance);
		
		return predictions;
	}

	/**
	 * Gets the distance metric considered by the classifier.
	 *
	 * @return The distance metric.
	 */
	public DistanceFunction getMetric() {
		return metric;
	}

	/**
	 * Sets the distance metric considered by the classifier.
	 *
	 * @param metric The new distance metric.
	 */
	public void setMetric(DistanceFunction metric) {
		this.metric = (DistanceFunction_MIMLWrapper)metric;
	}	

	public MultiLabelKNN getClassifier() {
		return classifier;
	}

	public void setClassifier(MultiLabelKNN classifier) {
		this.classifier = classifier;
	}	

	/**
	 * Gets the number of neigbors considered by the classifier.
	 *
	 * @return the number of neigbors
	 */
	public int getNumOfNeighbors() {
		return numOfNeighbors;
	}

	/**
	 * Sets the number of neigbors considered by the classifier.
	 *
	 * @param numOfNeighbors the new number of neigbors
	 */
	public void setnumOfNeighbors(int numOfNeighbors) {
		this.numOfNeighbors = numOfNeighbors;
	}
}
