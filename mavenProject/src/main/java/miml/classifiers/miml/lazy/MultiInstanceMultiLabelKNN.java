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
import miml.core.distance.HausdorffDistance;
import miml.core.distance.IDistance;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MultiLabelKNN;
import weka.core.DistanceFunction;

/** Wrapper for class MultiLabelKNN of Mulan to work with MIML data */
public abstract class MultiInstanceMultiLabelKNN extends MIMLClassifier {

	/**
	 * For serialization.
	 */
	private static final long serialVersionUID = 1L;

	/** Number of neighbours used in the k-nearest neighbor algorithm. */
	protected int numOfNeighbours;

	/** Metric for measure the distance between bags. */
	protected MIMLDistanceFunction metric;

	/** Mulan MultiLabelKNN classifier. */
	protected MultiLabelKNN classifier;

	/**
	 * Constructor to initialize the classifier. It sets the numberOfNeighbours to
	 * 10
	 * 
	 * @param metric The metric used by the algorithm to measure the distance
	 *               between bags.
	 * 
	 */
	public MultiInstanceMultiLabelKNN(MIMLDistanceFunction metric) {
		this.metric = metric;
		this.numOfNeighbours = 10;
	}

	/**
	 * Constructor to initialize the classifier. It sets the numOfNeighbours to 10
	 * 
	 * @param metric          The metric used by the algorithm to measure the
	 *                        distance between bags.
	 * @param numOfNeighbours The number of neighbours.
	 */
	public MultiInstanceMultiLabelKNN(MIMLDistanceFunction metric, int numOfNeighbours) {
		this.metric = metric;
		this.numOfNeighbours = numOfNeighbours;
	}

	/**
	 * No-arg constructor for xml configuration
	 */
	public MultiInstanceMultiLabelKNN() {
	}

	@SuppressWarnings("unchecked")
	@Override
	public void configure(Configuration configuration) {

		this.numOfNeighbours = configuration.getInt("numOfNeighbours", 10);
		try {
			// Get the name of the metric class
			String metricName = configuration.getString("metric[@name]", "miml.core.distance.AverageHausdorff");
			// Instance class
			Class<? extends IDistance> metricClass = (Class<? extends IDistance>) Class.forName(metricName);

			// this.metric = new DistanceFunction_MIMLWrapper(metricClass.newInstance());
			// //Java 8
			this.metric = new MIMLDistanceFunction(metricClass.getDeclaredConstructor().newInstance()); // Java 9
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}

	}

	@Override
	protected void buildInternal(MIMLInstances trainingSet) throws Exception {
		if (classifier == null)
			throw new Exception("The MultiLabelKNN classifier is null.");

		IDistance m = metric.getMetric();
		((HausdorffDistance) m).setInstances(trainingSet);

		classifier.setDfunc(metric);
		classifier.build(trainingSet.getMLDataSet());
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(MIMLBag instance) throws Exception, InvalidDataException {

		IDistance m = metric.getMetric();
		((HausdorffDistance) m).update(instance);

		MultiLabelOutput predictions = classifier.makePrediction(instance);

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
		this.metric = (MIMLDistanceFunction) metric;
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
	public int getNumOfNeighbours() {
		return numOfNeighbours;
	}

	/**
	 * Sets the number of neigbors considered by the classifier.
	 *
	 * @param numOfNeighbours the new number of neigbors
	 */
	public void setnumOfNeighbours(int numOfNeighbours) {
		this.numOfNeighbours = numOfNeighbours;
	}
}
