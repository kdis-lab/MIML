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

import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

import org.apache.commons.configuration2.Configuration;

import miml.classifiers.miml.MIMLClassifier;
import miml.core.distance.IDistance;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;
import mulan.core.ArgumentNullException;
import weka.core.matrix.Matrix;
import weka.core.matrix.SingularValueDecomposition;

/**
 * <p>
 * Class implementing the MIMLkNN algorithm for MIML data. For more information,
 * see <em>Zhang, M. L. (2010, October). A k-nearest neighbor based
 * multi-instance multi-label learning algorithm. In 2010 22nd IEEE
 * International Conference on Tools with Artificial Intelligence (Vol.2, pp.
 * 207-212). IEEE.</em>
 * </p>
 * 
 * @author Alvaro A. Belmonte
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20180608
 */
public class MIMLkNN extends MIMLClassifier {

	/** Generated Serial version UID. */
	private static final long serialVersionUID = 1L;

	/** Number of citers. */
	protected int nCiters = 1;

	/** Number of references. */
	protected int nReferences = 1;

	/** Metric for measure the distance between bags. */
	protected IDistance metric;

	/** MIML data. */
	protected MIMLInstances dataset;

	/** Dataset size (number of bags). */
	int dSize;

	/** Distance matrix between dataset's instances. */
	protected double[][] distanceMatrix;

	/** Instances' references matrix. */
	protected int[][] refMatrix;

	/** Weights matrix. */
	protected double[][] weightsMatrix;

	/** The t matrix. */
	protected double[][] tMatrix;

	/** The phi matrix. */
	protected double[][] phiMatrix;

	/**
	 * Basic constructor to initialize the classifier.
	 *
	 * @param nReferences The number of references considered by the algorithm.
	 * @param nCiters     The number of citers considered by the algorithm.
	 * @param metric         The metric used by the algorithm to measure the
	 *                       distance.
	 */
	public MIMLkNN(int nReferences, int nCiters, IDistance metric) {
		this.nCiters = nCiters;
		this.nReferences = nReferences;
		this.metric = metric;
	}

	/**
	 * Instantiates a new MIMLkNN with values by default except distance metric.
	 *
	 * @param metric The metric used by the algorithm to measure the distance.
	 */
	public MIMLkNN(IDistance metric) {
		this.metric = metric;
	}

	/**
	 * No-argument constructor for xml configuration.
	 */
	public MIMLkNN() {
	}

	/**
	 * @see miml.classifiers.miml.MIMLClassifier#buildInternal
	 */
	@Override
	protected void buildInternal(MIMLInstances trainingSet) throws Exception {
		if (trainingSet == null) {
			throw new ArgumentNullException("trainingSet");
		}

		this.dataset = trainingSet;
		dSize = trainingSet.getNumBags();

		// Change nReferences if its necessary
		if (dSize <= nReferences)
			nReferences = dSize - 1;

		// Initialize matrices
		tMatrix = new double[dSize][numLabels];
		phiMatrix = new double[dSize][numLabels];

		calculateDatasetDistances();
		calculateReferenceMatrix();

		for (int i = 0; i < dSize; ++i) {
			Integer[] neighbors = getUnionNeighbors(i);
			// Update matrices
			phiMatrix[i] = calculateRecordLabel(neighbors).clone();
			tMatrix[i] = getBagLabels(i).clone();
		}

		weightsMatrix = getWeightsMatrix();

	}

	/*
	 * (non-Javadoc)
	 * @see miml.classifiers.miml.MIMLClassifier#makePredictionInternal(miml.data.MIMLBag)
	 */
	@Override
	protected MultiLabelOutput makePredictionInternal(MIMLBag instance) throws Exception, InvalidDataException {

		// Create a new distances matrix
		double[][] distanceMatrixCopy = distanceMatrix.clone();
		distanceMatrix = new double[dSize + 1][dSize + 1];

		for (int i = 0; i < dSize; ++i) {
			// Fill distance matrix with previous values
			System.arraycopy(distanceMatrixCopy[i], 0, distanceMatrix[i], 0, dSize);
			// Update distance matrix with the new bag's distances
			double distance = metric.distance(instance, dataset.getBag(i));
			distanceMatrix[i][dSize] = distance;
			distanceMatrix[dSize][i] = distance;
		}

		// Update dSize to calculate references matrix
		dSize++;
		calculateReferenceMatrix();
		// Restore dSize value
		dSize--;

		Integer[] neighbors = getUnionNeighbors(dSize);
		double[] recordLabel = calculateRecordLabel(neighbors);

		double[] confidences = new double[numLabels];
		boolean[] predictions = new boolean[numLabels];

		// Apply linear classifier to each label
		for (int i = 0; i < numLabels; ++i) {
			double[] column = new double[numLabels];

			// Get column of weights
			for (int j = 0; j < numLabels; ++j)
				column[j] = weightsMatrix[j][i];

			boolean decision = linearClassifier(column, recordLabel);
			predictions[i] = decision;
			confidences[i] = (decision) ? 1.0 : 0.0;
		}

		MultiLabelOutput finalDecision = new MultiLabelOutput(predictions, confidences);
		// Restore original distance matrix
		distanceMatrix = distanceMatrixCopy.clone();

		return finalDecision;
	}

	/**
	 * Calculate the distances matrix of current data set with the metric assigned.
	 *
	 * @throws Exception The exception.
	 */
	protected void calculateDatasetDistances() throws Exception {

		distanceMatrix = new double[dSize][dSize];
		double distance;

		for (int i = 0; i < dSize; ++i) {

			MIMLBag first = dataset.getBag(i);
			for (int j = i; j < dSize; ++j) {
				MIMLBag second = dataset.getBag(j);
				distance = metric.distance(first, second);
				distanceMatrix[i][j] = distance;
				distanceMatrix[j][i] = distance;
			}
		}
	}

	/**
	 * Calculate the references matrix.
	 *
	 * @throws Exception the exception
	 */
	protected void calculateReferenceMatrix() throws Exception {

		refMatrix = new int[dSize][dSize];

		for (int i = 0; i < dSize; ++i) {

			int[] references = calculateBagReferences(i);

			for (int j = 0; j < references.length; ++j)
				refMatrix[i][references[j]] = 1;
		}
	}

	/**
	 * Calculate the references of a bag specified by its index. It's necessary
	 * calculate the distance matrix previously.
	 *
	 * @param indexBag The index bag.
	 * @return The references' indices of the bag.
	 * @throws Exception A exception.
	 */
	protected int[] calculateBagReferences(int indexBag) throws Exception {
		// Nearest neighbors of the selected bag
		int[] nearestNeighbors = new int[nReferences];
		// Store indices in priority queue, sorted by distance to selected bag
		PriorityQueue<Integer> pq = new PriorityQueue<Integer>(dSize,
				(a, b) -> Double.compare(distanceMatrix[indexBag][a], distanceMatrix[indexBag][b]));

		for (int i = 0; i < dSize; ++i) {
			if (i != indexBag)
				pq.add(i);
		}
		// Get the R (nReferences) nearest neighbors
		for (int i = 0; i < nReferences; ++i)
			nearestNeighbors[i] = pq.poll();

		return nearestNeighbors;
	}

	/**
	 * Gets the references of a specified bag.
	 *
	 * @param indexBag The index bag.
	 * 
	 * @return The bag's references.
	 */
	protected int[] getReferences(int indexBag) {

		int[] references = new int[nReferences];
		int idx = 0;

		for (int i = 0; i < dSize; ++i) {
			if (refMatrix[indexBag][i] == 1) {
				references[idx] = i;
				idx++;
			}
		}
		return references;
	}

	/**
	 * Calculate and return the citers of a bag specified by its index. It's
	 * necessary calculate the distance matrix first.
	 *
	 * @param indexBag The index bag.
	 * 
	 * @return The bag's citers.
	 */
	protected int[] getCiters(int indexBag) {

		PriorityQueue<Integer> pq = new PriorityQueue<Integer>(nReferences,
				(a, b) -> Double.compare(distanceMatrix[indexBag][a], distanceMatrix[indexBag][b]));

		for (int i = 0; i < dSize; ++i)
			if (refMatrix[i][indexBag] == 1)
				pq.add(i);

		int citers = (nCiters < pq.size()) ? nCiters : pq.size();
		// Nearest citers of the selected bag
		int[] nearestCiters = new int[citers];
		// Get the C (nCiters or pq.size()) nearest citers
		for (int i = 0; i < citers; ++i)
			nearestCiters[i] = pq.poll();

		return nearestCiters;
	}

	/**
	 * Gets the union of references and citers (without repetitions) of the bag
	 * specified.
	 *
	 * @param indexBag The index bag.
	 * 
	 * @return Ihe union of references and citers.
	 */
	protected Integer[] getUnionNeighbors(int indexBag) {

		int[] references = getReferences(indexBag);
		int[] citers = getCiters(indexBag);

		// Union references and citers sets
		Set<Integer> set = new HashSet<Integer>();

		for (int j = 0; j < references.length; j++)
			set.add(references[j]);
		for (int j = 0; j < citers.length; j++)
			set.add(citers[j]);

		Integer[] union = set.toArray(new Integer[set.size()]);
		return union;
	}

	/**
	 * Calculate the number of times each label appears in the bag's neighborhood.
	 *
	 * @param indices The neighboor's indices.
	 * 
	 * @return The labels' record.
	 */
	protected double[] calculateRecordLabel(Integer[] indices) {

		double[] labelCount = new double[numLabels];

		for (int i = 0; i < indices.length; ++i) {
			for (int j = 0; j < numLabels; ++j) {
				if (dataset.getDataSet().instance(indices[i]).stringValue(labelIndices[j]).equals("1"))
					labelCount[j]++;
			}
		}
		return labelCount;
	}

	/**
	 * Gets the labels of specified bag.
	 *
	 * @param bagIndex The bag index.
	 * 
	 * @return The bag labels.
	 */
	protected double[] getBagLabels(int bagIndex) {

		double[] labels = new double[numLabels];

		for (int i = 0; i < numLabels; ++i) {
			if (dataset.getDataSet().instance(bagIndex).stringValue(labelIndices[i]).equals("1"))
				labels[i] = 1;
			else
				labels[i] = -1;
		}
		return labels;
	}

	/**
	 * Calculate the weights matrix used for prediction.
	 *
	 * @return The weights matrix.
	 */
	protected double[][] getWeightsMatrix() {

		Matrix tMatrix = new Matrix(this.tMatrix);
		Matrix phiMatrix = new Matrix(this.phiMatrix);
		Matrix phiMatrixT = phiMatrix.transpose();

		Matrix A = phiMatrixT.times(phiMatrix);
		Matrix B = phiMatrixT.times(tMatrix);

		SingularValueDecomposition svd = A.svd();
		Matrix S = svd.getS();
		Matrix U = svd.getU();
		Matrix V = svd.getV();

		double[][] sDouble = S.getArray();
		double value;
		double threshold = 10e-10;

		for (int i = 0; i < sDouble[0].length; ++i) {
			value = sDouble[i][i];
			if (value < threshold)
				sDouble[i][i] = 0;
			else
				sDouble[i][i] = 1.0 / value;
		}

		S = new Matrix(sDouble);
		Matrix inverseA = V.times(S).times(U.transpose());
		// inverseA = inverseA.times(U.transpose());

		Matrix solution = inverseA.times(B);

		return solution.getArrayCopy();

	}

	/**
	 * Decide if a bag belong to a specified label.
	 *
	 * @param weights The weights correspondent to each label.
	 * 
	 * @param record  The labels' record of bag's neighbor to be predicted.
	 * 
	 * @return True, if belong, false if not.
	 */
	protected boolean linearClassifier(double[] weights, double[] record) {

		double decision = 0.0;
		// Multiply element by element
		for (int i = 0; i < numLabels; ++i)
			decision += weights[i] * record[i];

		return (decision > 0.0) ? true : false;
	}

	/**
	 * Returns the number of citers considered to estimate the class prediction of
	 * tests bags.
	 *
	 * @return The num citers.
	 */
	public int getNumCiters() {
		return nCiters;
	}

	/**
	 * Sets the number of citers considered to estimate the class prediction of
	 * tests bags.
	 *
	 * @param numCiters The new num citers.
	 */
	public void setNumCiters(int numCiters) {
		this.nCiters = numCiters;
	}

	/**
	 * Returns the number of references considered to estimate the class prediction
	 * of tests bags.
	 *
	 * @return The num references.
	 */
	public int getNumReferences() {
		return nReferences;
	}

	/**
	 * Sets the number of references considered to estimate the class prediction of
	 * tests bags.
	 *
	 * @param numReferences The new num references.
	 */
	public void setNumReferences(int numReferences) {
		this.nReferences = numReferences;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * core.IConfiguration#configure(org.apache.commons.configuration.Configuration)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void configure(Configuration configuration) {

		this.nReferences = configuration.getInt("nReferences", 1);
		this.nCiters = configuration.getInt("nCiters", 1);

		try {
			// Get the name of the metric class
			String metricName = configuration.getString("metric[@name]", "core.distance.AverageHausdorff");
			// Instance class
			Class<? extends IDistance> metricClass = (Class<? extends IDistance>) Class.forName(metricName);

			this.metric = metricClass.newInstance();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
}
