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
package miml.inProgress;

import org.apache.commons.configuration2.Configuration;

import miml.classifiers.miml.MIMLClassifier;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MLkNN;
import mulan.data.MultiLabelInstances;
import weka.core.DistanceFunction;
import weka.core.Instance;

/**
 * Class that implement Mulan MLkNN classifier to be used in this library as MIML classifier.
 * 
 * @author Alvaro A. Belmonte
 * @author Amelia Zafra
 * @author Eva Gigaja
 * @version 20190524
 */
public class MLkNNWrapper extends MIMLClassifier{

	/**
	 * Generated Serial version UID.
	 */
	private static final long serialVersionUID = -5918330265523388151L;

	/** Smooth factor. */
	protected double smooth = 1.0;
	
	/**Number of references. */
	protected int numReferences = 1;
	
	/** Metric for measure the distance between bags. */
	protected DistanceFunction metric;
	
	/** MIML data to build the classifier. */
	protected MIMLInstances dataset;
	
	/** ML data converted from MIML data. */
	protected MultiLabelInstances datasetConverted;
	
	/** Mulan MLkNN classifier. */
	protected MLkNN classifier;

	/**
	 * Basic constructor to initialize the classifier.
	 *
	 * @param numReferences The number of references considered by the algorithm.
	 * @param smooth	     The smooth factor considered by the algorithm.
	 * @param metric         The metric used by the algorithm to measure the
	 *                       distance.
	 */
	public MLkNNWrapper(int numReferences, double smooth, DistanceFunction metric) {
		this.numReferences = numReferences;
		this.smooth = smooth;
		this.metric = metric;
		this.classifier = new MLkNN(numReferences, smooth);
	}
	
	/**
	 * Instantiates a new MIMlkNN with values by default except distance metric.
	 *
	 * @param metric The metric used by the algorithm to measure the distance.
	 */
	public MLkNNWrapper(DistanceFunction metric) {
		this.metric = metric;
		this.classifier = new MLkNN(numReferences, smooth);
	}

	/**
	 *  No-arg constructor for xml configuration. 
	*/
	public MLkNNWrapper() {
	}
	
	/*
	 * (non-Javadoc)
	 * @see miml.classifiers.miml.MIMLClassifier#buildInternal(miml.data.MIMLInstances)
	 */
	@Override
	protected void buildInternal(MIMLInstances trainingSet) throws Exception {
		
		dataset = trainingSet;
		datasetConverted = dataset.getMLDataSet();
		classifier.setDfunc(metric);		
		classifier.build(datasetConverted);
	}

	/*
	 * (non-Javadoc)
	 * @see miml.classifiers.miml.MIMLClassifier#makePredictionInternal(miml.data.MIMLBag)
	 */
	@Override
	protected MultiLabelOutput makePredictionInternal(MIMLBag instance) throws Exception, InvalidDataException {
		
		Instance bagAsInstance = instance;
		MultiLabelOutput predictions = classifier.makePrediction(bagAsInstance);
		
		return predictions;
	}
	
	/*
	 * (non-Javadoc)
	 * @see miml.core.IConfiguration#configure(org.apache.commons.configuration2.Configuration)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void configure(Configuration configuration) {
		
		this.numReferences = configuration.getInt("numReferences");
		this.smooth = configuration.getDouble("smooth");
		
		try {
			//Get the name of the metric class
			String metricName = configuration.getString("metric[@name]");
			//Instance class
			Class<? extends DistanceFunction> metricClass = 
					(Class <? extends DistanceFunction>) Class.forName(metricName);
			
			this.metric = metricClass.newInstance();
		}
		catch(Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
		this.classifier = new MLkNN(numReferences, smooth);
	}
	
	/**
	 * Gets the smooth factor considered by the classifier.
	 *
	 * @return The smooth factor.
	 */
	public double getSmooth() {
		return smooth;
	}

	/**
	 * Sets the smooth factor considered by the classifier.
	 *
	 * @param smooth The new smooth factor.
	 */
	public void setSmooth(double smooth) {
		this.smooth = smooth;
	}

	/**
	 * Gets the number of references considered by the classifier.
	 *
	 * @return The number of references.
	 */
	public int getNumReferences() {
		return numReferences;
	}

	/**
	 * Sets the number of references considered by the classifier.
	 *
	 * @param numReferences The new number of references.
	 */
	public void setNumReferences(int numReferences) {
		this.numReferences = numReferences;
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
		this.metric = metric;
	}

}