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
	protected int num_citers = 1;

	/** Number of references. */
	protected int num_references = 1;

	/** Metric for measure the distance between bags. */
	protected IDistance metric;

	/** MIML data. */
	protected MIMLInstances dataset;

	/** Dataset size (number of bags). */
	int d_size;

	/** Distance matrix between dataset's instances. */
	protected double[][] distance_matrix;

	/** Instances' references matrix. */
	protected int[][] ref_matrix;

	/** Weights matrix. */
	protected double[][] weights_matrix;

	/** The t matrix. */
	protected double[][] t_matrix;

	/** The phi matrix. */
	protected double[][] phi_matrix;

	/**
	 * Basic constructor to initialize the classifier.
	 *
	 * @param num_references The number of references considered by the algorithm.
	 * @param num_citers     The number of citers considered by the algorithm.
	 * @param metric         The metric used by the algorithm to measure the
	 *                       distance.
	 */
	public MIMLkNN(int num_references, int num_citers, IDistance metric) {
		this.num_citers = num_citers;
		this.num_references = num_references;
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
		d_size = trainingSet.getNumBags();

		// Change num_references if its necessary
		if (d_size <= num_references)
			num_references = d_size - 1;

		// Initialize matrices
		t_matrix = new double[d_size][numLabels];
		phi_matrix = new double[d_size][numLabels];

		calculateDatasetDistances();
		calculateReferenceMatrix();

		for (int i = 0; i < d_size; ++i) {
			Integer[] neighbours = getUnionNeighbours(i);
			// Update matrices
			phi_matrix[i] = calculateRecordLabel(neighbours).clone();
			t_matrix[i] = getBagLabels(i).clone();
		}

		weights_matrix = getWeightsMatrix();

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see miml.classifiers.miml.MIMLClassifier#makePredictionInternal(miml.data.
	 * MIMLBag)
	 */
	@Override
	protected MultiLabelOutput makePredictionInternal(MIMLBag instance) throws Exception, InvalidDataException {

		// Create a new distances matrix
		double[][] distanceMatrixCopy = distance_matrix.clone();
		distance_matrix = new double[d_size + 1][d_size + 1];

		for (int i = 0; i < d_size; ++i) {
			// Fill distance matrix with previous values
			System.arraycopy(distanceMatrixCopy[i], 0, distance_matrix[i], 0, d_size);
			// Update distance matrix with the new bag's distances
			double distance = metric.distance(instance, dataset.getBag(i));
			distance_matrix[i][d_size] = distance;
			distance_matrix[d_size][i] = distance;
		}

		// Update d_size to calculate references matrix
		d_size++;
		calculateReferenceMatrix();
		// Restore d_size value
		d_size--;

		Integer[] neighbours = getUnionNeighbours(d_size);
		double[] recordLabel = calculateRecordLabel(neighbours);

		double[] confidences = new double[numLabels];
		boolean[] predictions = new boolean[numLabels];

		// Apply linear classifier to each label
		for (int i = 0; i < numLabels; ++i) {
			double[] column = new double[numLabels];

			// Get column of weights
			for (int j = 0; j < numLabels; ++j)
				column[j] = weights_matrix[j][i];

			boolean decision = linearClassifier(column, recordLabel);
			predictions[i] = decision;
			confidences[i] = (decision) ? 1.0 : 0.0;
		}

		MultiLabelOutput finalDecision = new MultiLabelOutput(predictions, confidences);
		// Restore original distance matrix
		distance_matrix = distanceMatrixCopy.clone();

		return finalDecision;
	}

	/**
	 * Calculate the distances matrix of current data set with the metric assigned.
	 *
	 * @throws Exception The exception.
	 */
	protected void calculateDatasetDistances() throws Exception {

		distance_matrix = new double[d_size][d_size];
		double distance;

		for (int i = 0; i < d_size; ++i) {

			MIMLBag first = dataset.getBag(i);
			for (int j = i; j < d_size; ++j) {
				MIMLBag second = dataset.getBag(j);
				distance = metric.distance(first, second);
				distance_matrix[i][j] = distance;
				distance_matrix[j][i] = distance;
			}
		}
	}

	/**
	 * Calculate the references matrix.
	 *
	 * @throws Exception the exception
	 */
	protected void calculateReferenceMatrix() throws Exception {

		ref_matrix = new int[d_size][d_size];

		for (int i = 0; i < d_size; ++i) {

			int[] references = calculateBagReferences(i);

			for (int j = 0; j < references.length; ++j)
				ref_matrix[i][references[j]] = 1;
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
		// Nearest neighbours of the selected bag
		int[] nearestNeighbours = new int[num_references];
		// Store indices in priority queue, sorted by distance to selected bag
		PriorityQueue<Integer> pq = new PriorityQueue<Integer>(d_size,
				(a, b) -> Double.compare(distance_matrix[indexBag][a], distance_matrix[indexBag][b]));

		for (int i = 0; i < d_size; ++i) {
			if (i != indexBag)
				pq.add(i);
		}
		// Get the R (num_references) nearest neighbours
		for (int i = 0; i < num_references; ++i)
			nearestNeighbours[i] = pq.poll();

		return nearestNeighbours;
	}

	/**
	 * Gets the references of a specified bag.
	 *
	 * @param indexBag The index bag.
	 * 
	 * @return The bag's references.
	 */
	protected int[] getReferences(int indexBag) {

		int[] references = new int[num_references];
		int idx = 0;

		for (int i = 0; i < d_size; ++i) {
			if (ref_matrix[indexBag][i] == 1) {
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

		PriorityQueue<Integer> pq = new PriorityQueue<Integer>(num_references,
				(a, b) -> Double.compare(distance_matrix[indexBag][a], distance_matrix[indexBag][b]));

		for (int i = 0; i < d_size; ++i)
			if (ref_matrix[i][indexBag] == 1)
				pq.add(i);

		int citers = (num_citers < pq.size()) ? num_citers : pq.size();
		// Nearest citers of the selected bag
		int[] nearestCiters = new int[citers];
		// Get the C (num_citers or pq.size()) nearest citers
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
	protected Integer[] getUnionNeighbours(int indexBag) {

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

		Matrix tMatrix = new Matrix(t_matrix);
		Matrix phiMatrix = new Matrix(phi_matrix);
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
	 * Classifier that determines the labels associated with an example. A linear
	 * classifier uses the label counting vector of the example and the weight
	 * vector corresponding to the label,
	 *
	 * 
	 * @param weights The weights correspondent to the label.
	 * @param record  The labels' record of bag's neighbor to be predicted.
	 * 
	 * @return True, if belong to a determinate class, false if not.
	 */
	protected boolean linearClassifier(double[] weights, double[] record) {

		double decision = 0.0;
		// Multiply element by element
		for (int i = 0; i < numLabels; ++i)
			decision += weights[i] * record[i];

		return (decision > 0.3) ? true : false;
	}

	/**
	 * Returns the number of citers considered to estimate the class prediction of
	 * tests bags.
	 *
	 * @return The number of citers.
	 */
	public int getNumCiters() {
		return num_citers;
	}

	/**
	 * Sets the number of citers considered to estimate the class prediction of
	 * tests bags.
	 *
	 * @param numCiters The new number of citers.
	 */
	public void setNumCiters(int numCiters) {
		this.num_citers = numCiters;
	}

	/**
	 * Returns the number of references considered to estimate the class prediction
	 * of tests bags.
	 *
	 * @return The number of references.
	 */
	public int getNumReferences() {
		return num_references;
	}

	/**
	 * Sets the number of references considered to estimate the class prediction of
	 * tests bags.
	 *
	 * @param numReferences The new number of references.
	 */
	public void setNumReferences(int numReferences) {
		this.num_references = numReferences;
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

		this.num_references = configuration.getInt("nReferences", 1);
		this.num_citers = configuration.getInt("nCiters", 1);

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
