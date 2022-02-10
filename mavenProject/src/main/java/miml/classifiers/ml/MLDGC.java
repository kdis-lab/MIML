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
package miml.classifiers.ml;

import mulan.classifier.MultiLabelOutput;
import mulan.classifier.lazy.MultiLabelKNN;
import mulan.data.MultiLabelInstances;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.neighboursearch.LinearNNSearch;

/**
 * Implementation of MLDGC (Multi-Label Data Gravitation Model) algorithm. For
 * more information see: <em> Oscar Reyes, Carlos Morell, Sebastián Ventura
 * (2016). Effective lazy learning algorithm based on a data gravitation model
 * for multi-label learning. Information Sciences. Vol 340, issue C. </em>
 *
 * @author Eva Gigaja
 * @version 20210604
 */
public class MLDGC extends MultiLabelKNN {

	/** For serialization */
	private static final long serialVersionUID = -2053344207047059082L;

	/** Neighborhood-based Gravitation Coefficient for each training example */
	protected double NGC[] = null;

	/** Densities */
	protected double densities[] = null;

	/** Weights */
	protected double weights[] = null;

	/** Searching of neighborhood */
	protected LinearNNESearch elnn;

	/**
	 * Whether neighborhood is extended with all the neighbors with the same
	 * distance. The default value is false.
	 */
	boolean extNeigh = false;

	/** Values used to normalize weights */
	protected double weight_max = Double.NEGATIVE_INFINITY;
	protected double weight_min = Double.POSITIVE_INFINITY;

	/**
	 * The default constructor. By default 10 neighbors and Euclidean distance.
	 */
	public MLDGC() {
		super();
		this.numOfNeighbors = 10;
		dfunc = new EuclideanDistance();
	}

	/**
	 * Constructor initializing the number of neighbors. By default Euclidean
	 * Distance.
	 *
	 * @param numOfNeighbors the number of neighbors
	 */
	public MLDGC(int numOfNeighbors) {
		super();
		this.numOfNeighbors = numOfNeighbors;
		dfunc = new EuclideanDistance();
	}

	/**
	 * Constructor initializing the number of neighbors and the distance function.
	 *
	 * @param numOfNeighbors the number of neighbors
	 * @param dfunc          distance function
	 */
	public MLDGC(int numOfNeighbors, DistanceFunction dfunc) {
		super();
		this.numOfNeighbors = numOfNeighbors;
		this.dfunc = dfunc;
	}

	@Override
	protected void buildInternal(MultiLabelInstances trainSet) throws Exception {

		super.buildInternal(trainSet);

		elnn = new LinearNNESearch(trainSet.getDataSet());
		elnn.setDistanceFunction(dfunc);
		elnn.setInstances(train);
		elnn.setMeasurePerformance(false);
		elnn.setSkipIdentical(true);

		labelIndices = trainSet.getLabelIndices();

		NGC = new double[trainSet.getNumInstances()];
		weights = new double[trainSet.getNumInstances()];
		densities = new double[trainSet.getNumInstances()];

		for (int i = 0; i < NGC.length; i++) {
			NGC[i] = 0.0;
			weights[i] = 0.0;
			densities[i] = 0.0;
		}

		for (int i = 0; i < trainSet.getNumInstances(); i++) {
			Instance instance = trainSet.getDataSet().instance(i);

			// compute knn for instance i
			Instances knn = elnn.kNearestNeighbours(instance, numOfNeighbors);

			// compute ngc for instance i
			computeWeightDensity(knn, instance, i);
		}

		// Normalize weights and compute NGC
		for (int i = 0; i < weights.length; i++) {
			weights[i] = weights[i] / (weight_max - weight_min);
			NGC[i] = Math.pow(densities[i], weights[i]);
		}

	}

	/**
	 * Computes the label distance between two instances.
	 *
	 * @param instance1 the first instance.
	 * @param instance2 the second instance.
	 * @return the label distance between two instances.
	 */
	protected double labelDistance(Instance instance1, Instance instance2) {
		double symmetricDifference = 0;
		int activeLabels = 0;
		for (int i = 0; i < this.labelIndices.length; i++) {
			if (instance1.value(labelIndices[i]) != instance2.value(labelIndices[i]))
				symmetricDifference++;
			if ((Utils.eq(instance1.value(labelIndices[i]), 1.0)) || (Utils.eq(instance2.value(labelIndices[i]), 1.0)))
				activeLabels++;
		}

		return symmetricDifference / labelIndices.length; // HammingLoss
		// return symmetricDifference / activeLabels; //Adjusted HammingLoss, considers
		// averaging by active labels instead all labels
	}

	/**
	 * Given a neighborhood and an instance, computes neighborhood-weight and
	 * neighborhood-density.
	 *
	 * @param knn      The neighborhood of the instance.
	 * @param instance The instance for which weight and density are computed.
	 * @param index    The index of the instance for which weight and density are
	 *                 computed.
	 */
	protected void computeWeightDensity(Instances knn, Instance instance, int index) {
		double weight = 1;
		double density = 0;

		double PdisY = 0;
		double PdisF = 0;
		double PdisY_disF = 0;

		int k;
		if (!extNeigh)
			k = numOfNeighbors;
		else
			k = knn.numInstances();

		for (int i = 0; i < k; i++) {
			Instance neighbor = knn.instance(i);
			double dl = labelDistance(instance, neighbor);
			double df = dfunc.distance(instance, neighbor);

			density += (1 - dl) / df;

			PdisY += dl;
			PdisF += df;
			PdisY_disF += dl * df;
		}

		// compute density
		density = 1 + density;

		// compute weight
		PdisY = PdisY / k;
		PdisF = PdisF / k;
		PdisY_disF = PdisY_disF / k;
		if ((PdisY == 0 || PdisY == 1))
			weight = 0;
		else
			weight = ((PdisY_disF * PdisF) / PdisY) - (((1 - PdisY_disF) * PdisF) / (1 - PdisY));

		if (weight_max < weight)
			weight_max = weight;
		if (weight_min > weight)
			weight_min = weight;

		weights[index] = weight;
		densities[index] = density;
	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {

		dfunc.update(instance);

		boolean[] bipartition = new boolean[labelIndices.length];
		double confidence[] = new double[labelIndices.length];

		Instances knn = elnn.kNearestNeighbours(instance, numOfNeighbors);
		int indices[] = elnn.kNearestNeighboursIndices(instance, numOfNeighbors);

		int k;
		if (!extNeigh)
			k = numOfNeighbors;
		else
			k = knn.numInstances();

		double gforce[] = new double[k];
		for (int index = 0, i = 0; i < k; i++) {
			Instance particle = knn.instance(i);
			index = indices[i];
			double distance = dfunc.distance(instance, particle);
			gforce[i] = NGC[index] / Math.pow(distance, 2);
		}

		for (int l = 0; l < bipartition.length; l++) {
			double positiveGF = 0.0;
			double negativeGF = 0.0;

			// computes positiveGF and negativeGF
			for (int i = 0; i < k; i++) {
				Instance neighbor = knn.instance(i);
				if (Utils.eq(neighbor.value(labelIndices[l]), 1.0))
					positiveGF += gforce[i];
				else
					negativeGF += gforce[i];
			}

			// computes bipartition and confidence
			if (positiveGF > negativeGF)
				bipartition[l] = true;
			else
				bipartition[l] = false;

			confidence[l] = positiveGF / (positiveGF + negativeGF);
		}

		MultiLabelOutput output = new MultiLabelOutput(bipartition, confidence);

		return output;
	}

	/**
	 * Gets the value of the property isExtNeigh.
	 * 
	 * @return the value of the property isExtNeigh.
	 */
	public boolean isExtNeigh() {
		return extNeigh;
	}

	/**
	 * Sets the value of the property isExtNeigh.
	 * 
	 * @param extNeigh the value to be set.
	 */
	public void setExtNeigh(boolean extNeigh) {
		this.extNeigh = extNeigh;
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}

	class LinearNNESearch extends LinearNNSearch {

		/** For serialization */
		private static final long serialVersionUID = 1L;

		public LinearNNESearch(Instances insts) throws Exception {
			super(insts);
		}

		public int[] kNearestNeighboursIndices(Instance target, int kNN) throws Exception {

			boolean print = false;

			if (m_Stats != null)
				m_Stats.searchStart();

			MyHeap heap = new MyHeap(kNN);
			double distance;
			int firstkNN = 0;
			for (int i = 0; i < m_Instances.numInstances(); i++) {
				if (target == m_Instances.instance(i)) // for hold-one-out cross-validation
					continue;
				if (m_Stats != null)
					m_Stats.incrPointCount();
				if (firstkNN < kNN) {
					if (print)
						System.out.println("K(a): " + (heap.size() + heap.noOfKthNearest()));
					distance = m_DistanceFunction.distance(target, m_Instances.instance(i), Double.POSITIVE_INFINITY,
							m_Stats);
					if (distance == 0.0 && m_SkipIdentical)
						if (i < m_Instances.numInstances() - 1)
							continue;
						else
							heap.put(i, distance);
					heap.put(i, distance);
					firstkNN++;
				} else {
					MyHeapElement temp = heap.peek();
					if (print)
						System.out.println("K(b): " + (heap.size() + heap.noOfKthNearest()));
					distance = m_DistanceFunction.distance(target, m_Instances.instance(i), temp.distance, m_Stats);
					if (distance == 0.0 && m_SkipIdentical)
						continue;
					if (distance < temp.distance) {
						heap.putBySubstitute(i, distance);
					} else if (distance == temp.distance) {
						heap.putKthNearest(i, distance);
					}
				}
			}

			Instances neighbours = new Instances(m_Instances, (heap.size() + heap.noOfKthNearest()));
			m_Distances = new double[heap.size() + heap.noOfKthNearest()];
			int[] indices = new int[heap.size() + heap.noOfKthNearest()];
			int i = 1;
			MyHeapElement h;
			while (heap.noOfKthNearest() > 0) {
				h = heap.getKthNearest();
				indices[indices.length - i] = h.index;
				m_Distances[indices.length - i] = h.distance;
				i++;
			}
			while (heap.size() > 0) {
				h = heap.get();
				indices[indices.length - i] = h.index;
				m_Distances[indices.length - i] = h.distance;
				i++;
			}

			m_DistanceFunction.postProcessDistances(m_Distances);

			for (int k = 0; k < indices.length; k++) {
				neighbours.add(m_Instances.instance(indices[k]));
			}

			if (m_Stats != null)
				m_Stats.searchFinish();

			return indices;
		}

	}

}
