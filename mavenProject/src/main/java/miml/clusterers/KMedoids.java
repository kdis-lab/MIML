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

package miml.clusterers;

import java.util.Arrays;
import java.util.Random;

import miml.core.distance.IDistance;
import miml.core.distance.MaximalHausdorff;
import weka.clusterers.RandomizableClusterer;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Class implementing the PAM (Partitioning Around Medoids) approximation [1] to
 * kMedoids for multi-instance data.
 * 
 * [1] <em> Kaufman, L. and Rousseeuw, P.J. (1990). Partitioning Around Medoids
 * (Program PAM). In Finding Groups in Data (eds L. Kaufman and P.J. Rousseeuw).
 * https://doi.org/10.1002/9780470316801.ch2 </em>
 * 
 * 
 * @author Eva Gibaja
 * @version 20230412
 */
public class KMedoids extends RandomizableClusterer implements weka.clusterers.Clusterer {

	/** For serialization. */
	private static final long serialVersionUID = -6814942755920034118L;

	/** Distance function. By default MaximalHausdorff distance is used */
	protected IDistance metric;

	/** Number of clusters to generate. By default 10 clusters. */
	protected int numClusters = 10;

	/** Number of instances in the dataset. */
	protected int numInstances;

	/**
	 * The maximum number of iterations the algorithm is allowed to run. By default
	 * 100 iterations
	 */
	protected int maxIterations = 100;

	/**
	 * The medoid indices. Element k contains the index of the instance being the k
	 * medoid, a value in (0, numInstances).
	 */
	protected int[] medoidIndices;

	/** The medoid instances. Element k contains the instance being the k medoid */
	protected Instance[] medoidInstances;

	/**
	 * The assignment of instances to medoids. Element i contains the number of
	 * medoid assigned to instance i, a value in (0, nClusters-1).
	 */
	protected int[] clusterAssignment;

	/** Distance between instances. */
	protected double distancesMatrix[][];

	/**
	 * Whether the metric is maximized o minimized. By default the metric is
	 * minimized.
	 */
	protected boolean minimize = true;

	/**
	 * Whether the initialization of medoids is random o applying the BUILD method
	 * of PAM algorithm. By default random initialization is performed.
	 */
	protected boolean randomInitialization = true;

	/** Final cost of the clustering configuration. */
	protected double configurationCost;

	/** Final number of iterations to perform clustering. */
	protected double numIterations;

	/**
	 * Creates a new instance of the k-medoids algorithm with default parameters.
	 * 
	 * @throws Exception To be handled in an upper level.
	 */
	public KMedoids() throws Exception {
		this(10, 100, new MaximalHausdorff());
	}

	/**
	 * Creates a new instance of the k-medoids algorithm with with the specified
	 * parameters.
	 * 
	 * @param numClusters The number of clusters.
	 * @throws Exception To be handled in an upper level.
	 */
	public KMedoids(int numClusters) throws Exception {
		this(numClusters, 100, new MaximalHausdorff());
	}

	/**
	 * Creates a new instance of the k-medoids algorithm with the specified dist
	 * measure.
	 * 
	 * @param metric The distance metric to use for measuring the distance between
	 *               instances.
	 * @throws Exception To be handled in an upper level.
	 */
	public KMedoids(IDistance metric) throws Exception {
		this(10, 100, metric);
	}

	/**
	 * Creates a new instance of the k-medoids algorithm with the specified
	 * parameters.
	 * 
	 * @param numClusters The number of clusters to generate.
	 * @param metric      The distance metric to use for measuring the distance
	 *                    between instances.
	 * @throws Exception To be handled in an upper level.
	 */

	public KMedoids(int numClusters, IDistance metric) throws Exception {
		this(numClusters, 100, metric);
	}

	/**
	 * Creates a new instance of the k-medoids algorithm with the specified
	 * parameters.
	 * 
	 * @param numClusters   The number of clusters to generate.
	 * @param maxIterations The maximum number of iteration the algorithm is allowed
	 *                      to run.
	 * @param metric        The distance metric to use for measuring the distance
	 *                      between instances.
	 * @throws Exception To be handled in an upper level.
	 */
	public KMedoids(int numClusters, int maxIterations, IDistance metric) throws Exception {
		super();
		this.metric = metric;
		this.maxIterations = maxIterations;
		this.numClusters = numClusters;
	}

	/**
	 * Allows to maximize or minimize the metric according to the value of minimize
	 * property.
	 * 
	 * @param metricValue1 A metric value.
	 * @param metricValue2 Another metric value.
	 * @return If minimize==true it returns metricValue1&lt;=metricValue2 other case
	 *         it returns metricValue1&gt;=metricValue2.
	 */
	protected boolean compare(double metricValue1, double metricValue2) {
		if (minimize)
			return metricValue1 <= metricValue2;
		else
			return metricValue1 >= metricValue2;
	}

	/** Performs a random initialization of medoids. */
	protected void randomInitialization() {
		Random rg;

		rg = new Random(this.getSeed());

		// Random initialization of medoids
		for (int k = 0; k < numClusters; k++) {
			int random = rg.nextInt(numInstances);
			while (isMedoid(random)) // to avoid duplicated medoid
				random = rg.nextInt(numInstances);
			medoidIndices[k] = random;
		}
	}

	/**
	 * Performs an initialization of medoids based on the BUILD step of PAM
	 * algorithm.
	 */
	protected void buildInitialization() {

		// Computes the sum of distances of each instance to the rest
		double sumDistances[] = new double[numInstances];

		int bestIndex = 0;
		for (int i = 0; i < numInstances; i++) {
			sumDistances[i] = 0;
			for (int j = 0; j < numInstances; j++) {
				if (j != i)
					sumDistances[i] += distancesMatrix[i][j];
			}
			if (compare(sumDistances[i], sumDistances[bestIndex]))
				bestIndex = i;
		}

		// The best medoid is the one with minimal sum of the distances to all other
		// objects
		int k = 0;
		medoidIndices[k] = bestIndex;

		for (k = 1; k < numClusters; k++) {

			double gain[] = new double[numInstances];
			Arrays.fill(gain, 0);

			for (int i = 0; i < numInstances; i++) {
				if (!isMedoid(i)) {

					for (int j = 0; j < numInstances; j++) {
						if (!isMedoid(j) && j != i) {

							// Dj is the mimimal distance of j to the centroids
							double Dj = distancesMatrix[j][medoidIndices[0]];
							for (int c = 1; c < k; c++)
								if (compare(Dj, distancesMatrix[j][c])) {
									Dj = distancesMatrix[j][c];
								}

							if (!(compare(Dj, distancesMatrix[i][j]))) {
								gain[i] += Math.abs(Dj - distancesMatrix[i][j]);
								;
							}

						} // if !isMedoid(j)
					} // for j
				} // if !isMedoid(i)
			} // for i

			// Select as medoid the instance i that maximizes gain
			bestIndex = 0;
			for (int i = 1; i < numInstances; i++) {
				if (!isMedoid(i) && gain[i] > gain[bestIndex])
					bestIndex = i;
			}
			medoidIndices[k] = bestIndex;

		} // for k

	}

	@Override
	public void buildClusterer(Instances data) throws Exception {

		numInstances = data.numInstances();
		if ((numClusters > numInstances)) {
			// throw new Exception("\nThe number of clusters must be less or equal to the
			// number of bags.");
			System.out.println("The number of clusters must be less or equal to the number of bags. Setting nClusters="
					+ numInstances);
			numClusters = numInstances;
		}
		if (numClusters < 2) {
			// throw new Exception("\nThe number of clusters must be at least 2.");
			System.out.println("\nThe number of clusters must be at least 2. Setting numClusters=2");
			numClusters = 2;
		}

		metric.setInstances(data);

		// Initialization of distance matrix

		distancesMatrix = new double[numInstances][numInstances];
		computeDistances(data);

		// BUILD STEP. Initialization of medoids.
		medoidIndices = new int[numClusters];
		for (int k = 0; k < numClusters; k++)
			medoidIndices[k] = -1;

		if (randomInitialization)
			randomInitialization();
		else
			buildInitialization();

		clusterAssignment = assignInstancesToMedoids(medoidIndices);
		double cost = computeCost(clusterAssignment);

		// SWAP STEP
		boolean change = true;
		int count = 0;
		while (change && count < maxIterations) {

			change = false;

			// for each medoid k
			for (int k = 0; k < medoidIndices.length; k++) {

				int oldMedoid = medoidIndices[k];

				// for each instance i not medoid
				for (int i = 0; i < data.numInstances(); i++) {

					if (!isMedoid(i)) {

						// Considers swaps the medoid and the instance
						medoidIndices[k] = i;
						int[] candidateAsignment = assignInstancesToMedoids(medoidIndices);
						double candidateCost = computeCost(candidateAsignment);

						if (compare(candidateCost, cost)) {
							// the configuration is better and the change is accepted
							clusterAssignment = candidateAsignment.clone();
							cost = candidateCost;
							change = true;
						} else {
							// the change is not accepted
							medoidIndices[k] = oldMedoid;
						}
					}

				} // for each instance i not medoid

			} // for each medoid k
			count++;
		}

		// Sets the array with the medoid instances
		medoidInstances = new Instance[numClusters];
		for (int k = 0; k < numClusters; k++)
			medoidInstances[k] = data.instance(medoidIndices[k]);

		this.configurationCost = cost;
		this.numIterations = count;
		// System.out.println("\nFinal configuration cost: " + cost + " Iterations: " +
		// count);
	}

	/**
	 * Computes distances between instances.
	 * 
	 * @param data The dataset.
	 * @throws Exception To be handled in an upper level.
	 */
	protected void computeDistances(Instances data) throws Exception {
		for (int i = 0; i < data.numInstances(); i++) {
			// Diagonal elements have 0 value
			distancesMatrix[i][i] = 0;

			Instance instanceA = data.instance(i);

			for (int j = i + 1; j < data.numInstances(); j++) {
				Instance instanceB = data.instance(j);
				double dist = metric.distance(instanceA, instanceB);
				distancesMatrix[i][j] = dist;
				distancesMatrix[j][i] = dist;
				// System.out.println("\nDistance[" + i + "][" + j + "]:"+dist);
			}
		}
	}

	/**
	 * Assign all instances from the data set to the medoids.
	 * 
	 * @param medoidIndices Candidate medoids.
	 * @return An array with the best cluster number for each instance in the data
	 *         set.
	 */
	protected int[] assignInstancesToMedoids(int[] medoidIndices) {

		// double cost = 0;
		clusterAssignment = new int[numInstances];

		for (int i = 0; i < numInstances; i++) {

			int index = medoidIndex(i);
			if (index >= 0) {
				// if i is medoid it is assigned to itself with 0 cost
				clusterAssignment[i] = index;
			} else {

				double bestDistance = distancesMatrix[i][medoidIndices[0]];
				int bestMedoidIndex = 0;

				for (int k = 1; k < medoidIndices.length; k++) {

					double auxDistance = distancesMatrix[i][medoidIndices[k]];
					if (compare(auxDistance, bestDistance)) {
						// System.out.println("\n\t"+auxDistance+"<="+bestDistance);
						bestDistance = auxDistance;
						bestMedoidIndex = k;
					}

				}
				clusterAssignment[i] = bestMedoidIndex;
				// cost += distances[i][medoidIndices[bestMedoidIndex]];
				// System.out.println("\nSumando:
				// "+bestDistance+"=="+distances[i][medoidIndices[assignment[i]]]+"=="+distances[i][medoidIndices[bestMedoidIndex]]);
			}
		}

		// double cost2=computeCost(assignment);
		// System.out.println("COST:"+cost+"=="+cost2);
		return clusterAssignment;
	}

	/**
	 * Computes the cost of a configuration.
	 * 
	 * @param assignment Array containing in element i the index of the medoid
	 *                   assigned to instance i.
	 * @return The sum of the distances to medoids of all instances.
	 */
	protected double computeCost(int[] assignment) {
		double cost = 0;
		for (int i = 0; i < assignment.length; i++) {
			cost += distancesMatrix[i][medoidIndices[assignment[i]]];
		}
		return cost;
	}

	/**
	 * Determines if an instance is being considered as medoid.
	 * 
	 * @param instanceIndex The index of the instance.
	 * @return A true value if the instance is being considered as medoid.
	 */
	protected boolean isMedoid(int instanceIndex) {

		for (int k = 0; k < medoidIndices.length; k++)
			if (medoidIndices[k] == instanceIndex)
				return true;

		return false;
	}

	/**
	 * Determines if an instance is being considered as medoid. If true, the index
	 * of the medoid is returned, a value in (0, nClusters-1)
	 * 
	 * @param instanceIndex The index of the instance.
	 * @return A true value if the instance is being considered as medoid.
	 */
	protected int medoidIndex(int instanceIndex) {
		for (int k = 0; k < medoidIndices.length; k++)
			if (medoidIndices[k] == instanceIndex)
				return k;

		return -1;
	}

	/**
	 * Returns the distance of an instance to each medoid.
	 * 
	 * @param instance An instance. It can be either an instance of the dataset or a
	 *                 new instance.
	 * @return The distance of the instance to each medoid.
	 * @throws Exception To be handled in an upper level.
	 */
	public double[] distanceToMedoids(Instance instance) throws Exception {

		metric.update(instance);

		double distances[] = new double[numClusters];

		for (int k = 0; k < numClusters; k++) {
			distances[k] = metric.distance(medoidInstances[k], instance);
		}
		return distances;
	}

	/**
	 * Returns the distance of an instance in the training dataset referenced by its
	 * index to each medoid.
	 * 
	 * @param index It must be a valid instance index in the dataset used for
	 *              clustering.
	 * @return The distance of the instance to each medoid.
	 * @throws Exception To be handled in an upper level.
	 */
	public double[] distanceToMedoids(int index) throws Exception {

		if ((index < 0) || (index > numInstances))
			throw new Exception(
					"The index must be >0 and <numInstances and the method has received an index of " + index);

		double distances[] = new double[numClusters];

		for (int k = 0; k < numClusters; k++) {
			distances[k] = distancesMatrix[index][medoidIndices[k]];

		}
		return distances;
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		// Computes distances to all medoids
		double distances[] = distanceToMedoids(instance);

		// Gets a value in (0, 1)
		double sumMinimize = 0;
		double sumMaximize = 0;

		// Computes the sum of the distances to all medoids
		for (int k = 0; k < numClusters; k++) {
			sumMinimize += (1.0 / distances[k]);
			sumMaximize += distances[k];
		}

		// Computes the distribution
		double distribution[] = new double[numClusters];
		for (int k = 0; k < numClusters; k++) {
			if (minimize) {
				distribution[k] = (1.0 / distances[k]) / sumMinimize;
			} else {
				distribution[k] = distances[k] / sumMaximize;
			}
		}
		return distribution;
	}

	@Override
	public int clusterInstance(Instance instance) throws Exception {

		double[] evaluation = distributionForInstance(instance);
		return Utils.maxIndex(evaluation);
	}

	// --------------------
	// GETTERS AND SETTERS
	// --------------------

	@Override
	public int numberOfClusters() throws Exception {
		return numClusters;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}

	/**
	 * Gets the medoids obtained after performing clustering.
	 * 
	 * @return An array of instances corresponding to medoids.
	 */
	public Instance[] getMedoidInstances() {
		return medoidInstances;
	}

	/**
	 * Gets the distance function used by clusterer.
	 * 
	 * @return The distance function used by clusterer.
	 */
	public IDistance getDistanceFunction() {
		return metric;
	}

	/**
	 * Sets the distance function to use for clustering. This method must be called
	 * before clustering.
	 * 
	 * @param distanceFunction The distance function used for clustering.
	 */
	public void setDistanceFunction(IDistance distanceFunction) {
		this.metric = distanceFunction;
	}

	/**
	 * Sets the number of clusters to perform clustering. This method must be called
	 * before clustering.
	 * 
	 * @param numClusters A number of clusters.
	 */
	public void setNumClusters(int numClusters) {
		this.numClusters = numClusters;
	}

	/**
	 * Gets the maximum number of iterations used by clusterer.
	 * 
	 * @return The maximum number of iterations.
	 */
	public int getMaxIterations() {
		return maxIterations;
	}

	/**
	 * Sets the maximum number of iterations for clustering. This method must be
	 * called before clustering.
	 * 
	 * @param maxIterations The maximum number of iterations for clustering.
	 */
	public void setMaxIterations(int maxIterations) {
		this.maxIterations = maxIterations;
	}

	/**
	 * Gets whether a random initialization of medoids or a initialization based on
	 * the BUILD step of PAM is considered for clustering.
	 * 
	 * @return A true value if a random initialization of medoids is performed and
	 *         false if the initialization is based on the build step of PAM
	 *         selecting as medoids, the instances that minimizes the sum of
	 *         distances to the rest.
	 */
	public boolean getRandomInitialization() {
		return randomInitialization;
	}

	/**
	 * Sets whether a random initialization of medoids or a initialization based on
	 * the BUILD step of PAM is considered for clustering. This method must be
	 * called before clustering.
	 * 
	 * @param randomInitialization If true a random initialization of medoids is
	 *                             performed. Otherwise the initialization is based
	 *                             on the build step of PAM selecting as medoids,
	 *                             the instances that minimizes the sum of distances
	 *                             to the rest.
	 */
	public void setRandomInitialization(boolean randomInitialization) {
		this.randomInitialization = randomInitialization;
	}

	/**
	 * Gets the assignment of instances to clusters. This method must be called
	 * after clustering.
	 * 
	 * @return An array. Element i contains a value in (0, numCusters-1), the
	 *         cluster number assigned to instance i.
	 */
	public int[] getAssignment() {
		return clusterAssignment;
	}

	/**
	 * Gets final the cost of the configuration after applying clustering. This
	 * method must be called after clustering.
	 * 
	 * @return The final cost of the clustering.
	 */
	public double getConfigurationCost() {
		return configurationCost;
	}

	/**
	 * Gets the number of iterations performed in the clustering process. This
	 * method must be called after clustering.
	 * 
	 * @return The number of iterations performed.
	 */
	public double getNumIterations() {
		return numIterations;
	}

	/**
	 * Returns a matrix the distances between all instances being distances[i][j]
	 * the distance between the instances with indices i and j.
	 * 
	 * @return double[][]
	 */

	public double[][] getDistances() {
		return distancesMatrix;
	}

}
