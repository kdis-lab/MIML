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
package miml.transformation.mimlTOml;

import java.util.ArrayList;

import miml.clusterers.KMedoids;
import miml.core.distance.IDistance;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class implementing the medoid-based transformation described in [1] to
 * transform an MIML problem to ML.
 * 
 * [1] <em> Zhou, Z. H., Zhang, M. L., Huang, S. J., &amp; Li, Y. F. (2012).
 * Multi-instance multi-label learning. Artificial Intelligence, 176(1),
 * 2291-2320. </em>
 * 
 * This class requires method transformDataset to have been executed before
 * executing transformInstance method.
 * 
 * @author Eva Gibaja
 * @version 20230412
 */
public class MedoidTransformation extends MIMLtoML {

	/** For serialization */
	private static final long serialVersionUID = 94921720184805609L;

	/** Clusterer. */
	protected KMedoids clusterer = null;

	/**
	 * True if the resulting transformed dataset will be normalized to (0,1) with
	 * min-max normalization. By default False. If a learning algorithm that uses a
	 * NormalizableDistance is going to be used after transformation, normalization
	 * is not needed.
	 */
	protected Boolean normalize = false;

	/**
	 * If it is different to -1 this value represent that the number of clusters
	 * will be a percentage of the number of bags of the dataset. For instance 0.2
	 * represents that the number of clusters is the 20% of the training bags, 0.45
	 * a 45%, and so on. If this value is -1 the number of clusters to consider is
	 * represented by numClusters property. If the number of clusters is not set
	 * neither by percentage nor by the numClusters property, it will be considered
	 * by default a 20% of the number of training bags in the dataset. If both the
	 * percentage and the numClusters are set, the percentage will be applied.
	 */
	protected double percentage = -1;

	/** The number of clusters for kmedoids. */
	protected int numClusters = -1;

	/** Whether the clustering step has been executed or not. */
	protected boolean clusteringDone = false;

	/** The seed for kmedoids clustering. By default 1. */
	protected int seed = 1;

	/**
	 * Constructor. Uses the same default number of clusters as MIMLSVM: 20% of
	 * number of bags
	 * 
	 * @param dataset MIMLInstances dataset.
	 * @throws Exception To be handled in an upper level.
	 */
	public MedoidTransformation(MIMLInstances dataset) throws Exception {

		this(dataset, new KMedoids(), false);
	}

	/**
	 * Constructor.
	 * 
	 * @param dataset   MIMLInstances dataset.
	 * @param kmedoids  An instance of kmedoids.
	 * @param normalize If true, the resulting transformed dataset will be
	 *                  normalized to (0,1) with min-max normalization. If a
	 *                  learning algorithm that uses a NormalizableDistance is going
	 *                  to be used, normalization is not needed.
	 * @throws Exception To be handled in an upper level.
	 */
	public MedoidTransformation(MIMLInstances dataset, KMedoids kmedoids, boolean normalize) throws Exception {

		super(dataset);
		this.clusterer = kmedoids;
		this.normalize = normalize;
	}

	/**
	 * Constructor.
	 * 
	 * @param dataset     MIMLInstances dataset.
	 * @param numClusters The number of clusters for kmedoids.
	 * @param metric      The distance function to be used by kmedoids.
	 * @throws Exception To be handled in an upper level.
	 */
	public MedoidTransformation(MIMLInstances dataset, IDistance metric, int numClusters) throws Exception {
		this(dataset, new KMedoids(numClusters, metric), false);
		setNumClusters(numClusters);
	}

	/**
	 * Constructor.
	 * 
	 * @param dataset     MIMLInstances dataset.
	 * @param numClusters number of clusters for kmedoids.
	 * @throws Exception To be handled in an upper level.
	 */
	public MedoidTransformation(MIMLInstances dataset, int numClusters) throws Exception {
		this(dataset, new KMedoids(numClusters), false);
		setNumClusters(numClusters);
	}

	/**
	 * Constructor.
	 * 
	 * @param dataset    MIMLInstances dataset.
	 * @param percentage The number of clusters for kmedoids as a percentage of the
	 *                   number of bags. It is a value in (0,1). For instance, 0.2
	 *                   is 20%.
	 * @throws Exception To be handled in an upper level.
	 */
	public MedoidTransformation(MIMLInstances dataset, double percentage) throws Exception {
		this(dataset, new KMedoids((int) (percentage * dataset.getNumBags())), false);
		this.percentage = percentage;
	}

	/**
	 * Constructor.
	 */
	public MedoidTransformation() {
		super();
	}

	protected void clusteringStep() throws Exception {
		if (clusterer == null) {
			clusterer = new KMedoids();
		}

		configureClusterer();

		System.out.println("Medoid Transformation\n\tPerforming kmedoids clustering to transform the dataset");
		clusterer.buildClusterer(dataset.getDataSet());
		clusteringDone = true;
		System.out.println(
				"\t" + clusterer.numberOfClusters() + " clusters in " + clusterer.getNumIterations() + " iterations");
		prepareTemplate();
		template.setRelationName(dataset.getDataSet().relationName() + "_medoid_transformation");
	}

	@Override
	public MultiLabelInstances transformDataset() throws Exception {

		// Clustering with kmedoids step
		clusteringStep();

		// Transformation step
		Instances newData = new Instances(template);
		int labelIndices[] = dataset.getLabelIndices();
		Instance newInst = new DenseInstance(newData.numAttributes());
		newInst.setDataset(newData); // Sets the reference to the dataset

		// For all bags in the dataset
		double nBags = dataset.getNumBags();
		int numClusters = clusterer.numberOfClusters();
		for (int i = 0; i < nBags; i++) {

			// retrieves a bag
			MIMLBag bag = dataset.getBag(i);

			// sets the bagLabel
			newInst.setValue(0, bag.value(0));

			// computes distances to medoids
			// double[] distances = kmedoids.distanceToMedoids(bag);
			double[] distances = clusterer.distanceToMedoids(i);

			// an attribute for medoid
			for (int k = 0, attIdx = 1; k < numClusters; k++, attIdx++) {
				newInst.setValue(attIdx, distances[k]);
			}

			// Copy label information into the dataset
			for (int j = 0; j < labelIndices.length; j++) {
				newInst.setValue(updatedLabelIndices[j], bag.value(labelIndices[j]));
			}
			newData.add(newInst);

		}

		if (normalize == true) {
			System.out.println("\t Performing min-max normalization on the transformed dataset.");
			return normalize(new MultiLabelInstances(newData, dataset.getLabelsMetaData()));
		} else
			return new MultiLabelInstances(newData, dataset.getLabelsMetaData());

	}

	@Override
	public MultiLabelInstances transformDataset(MIMLInstances dataset) throws Exception {

		MultiLabelInstances transformed = null;
		if (!clusteringDone) {
			this.dataset = dataset;
			transformed = transformDataset();
		} else {
			// To avoid a new clustering round if the clustering step was previously
			// performed
			Instances newData = new Instances(template);
			for (int i = 0; i < dataset.getNumBags(); i++) {
				MIMLBag bag = dataset.getBag(i);
				Instance transfInst = transformInstance(bag);
				newData.add(transfInst);
			}
			transformed = new MultiLabelInstances(newData, dataset.getLabelsMetaData());
		}
		return transformed;

	}

	/**
	 * Normalizes a multi-label dataset performing min-max normalization.
	 * 
	 * @param dataset The dataset to be normalized.
	 * @return Returns the normalized dataset as MultiLabelInstances.
	 * @throws Exception To be handled in an upper level.
	 */
	protected MultiLabelInstances normalize(MultiLabelInstances dataset) throws Exception {

		// 1. Computes statistics to perform normalization

		// number of attributes including the bagID attribute
		int nFeatures = dataset.getFeatureAttributes().size();
		double Max[] = new double[nFeatures];
		double Min[] = new double[nFeatures];
		double Range[] = new double[nFeatures];

		for (int i = 0; i < nFeatures; i++) {
			Max[i] = Double.NEGATIVE_INFINITY;
			Min[i] = Double.POSITIVE_INFINITY;
			Range[i] = 0;
		}

		boolean isNormalized = true;
		for (int i = 0; i < dataset.getNumInstances(); i++) {
			Instance instance = dataset.getDataSet().instance(i);
			// j=1 to ignore the bagId attribute
			for (int j = 1; j < nFeatures; j++) {

				if (instance.attribute(j).isNumeric()) {
					if (instance.value(j) < 0 || instance.value(j) > 1)
						isNormalized = false;
					if (instance.value(j) < Min[j])
						Min[j] = instance.value(j);
					if (instance.value(j) > Max[j])
						Max[j] = instance.value(j);
				}
			}

		}

		// j=1 to ignore the bagId attribute
		for (int i = 1; i < nFeatures; i++) {
			Range[i] = Max[i] - Min[i];
		}

		// 2. Normalizes the dataset
		if (isNormalized)
			return dataset;
		else {

			for (int i = 0; i < dataset.getNumInstances(); i++) {
				Instance instance = dataset.getDataSet().instance(i);

				// j=1 to ignore the bagId attribute
				for (int j = 1; j < nFeatures; j++) {

					double value = instance.value(j);

					// to avoid dividing by zero in case of a 0 range
					if (Double.compare(Min[j], Max[j]) != 0) {
						value = (value - Min[j]) / (Range[j]);

					} else {

						value = 1;
					}
					instance.setValue(j, value);
				}
			}
			return dataset;
		}

	}

	/**
	 * Returns the value of the property normalize.
	 * 
	 * @return The value of the property normalize.
	 */
	public Boolean getNormalize() {
		return normalize;
	}

	/**
	 * Sets the property normalized. If true, the resulting transformed multi-label
	 * dataset will be normalized after transformation.
	 * 
	 * @param normalize The value of the property to be set.
	 */
	public void setNormalize(Boolean normalize) {
		this.normalize = normalize;
	}

	@Override

	public Instance transformInstance(MIMLBag bag) throws Exception {

		if (!clusteringDone)
			throw new Exception(
					"The transformInstance method must be called after executing transformDataset that performs kmedoids clustering required by this kind of transformation.");

		int labelIndices[] = dataset.getLabelIndices();
		Instance newInst = new DenseInstance(template.numAttributes());

		// sets the bagLabel
		newInst.setDataset(bag.dataset()); // Sets the reference to the dataset
		newInst.setValue(0, bag.value(0));

		// computes distances to medoids, the bag could be either a bag in the clustered
		// dataset or a new and previously unseen bag.
		double[] distance = this.clusterer.distanceToMedoids(bag);

		// an attribute for medoid
		int numClusters = clusterer.numberOfClusters();
		for (int k = 0, attIdx = 1; k < numClusters; k++, attIdx++) {
			newInst.setValue(attIdx, distance[k]);
		}

		// Insert label information into the instance
		for (int j = 0; j < labelIndices.length; j++) {
			newInst.setValue(updatedLabelIndices[j], bag.value(labelIndices[j]));
		}

		return newInst;
	}

	@Override
	protected void prepareTemplate() throws Exception {
		int attrIndex = 0;

		ArrayList<Attribute> attributes = new ArrayList<Attribute>();

		// insert a bag label attribute at the beginning
		Attribute attr = dataset.getDataSet().attribute(0);
		attributes.add(attr);

		// Adds attributes for medoids
		int numClusters = clusterer.numberOfClusters();
		for (int k = 1; k <= numClusters; k++) {
			attr = new Attribute("distanceToMedoid_" + k);
			attributes.add(attr);
			attrIndex++;
		}

		// Insert labels as attributes in the dataset
		int labelIndices[] = dataset.getLabelIndices();
		updatedLabelIndices = new int[labelIndices.length];
		ArrayList<String> values = new ArrayList<String>(2);
		values.add("0");
		values.add("1");
		for (int i = 0; i < labelIndices.length; i++) {
			attr = new Attribute(dataset.getDataSet().attribute(labelIndices[i]).name(), values);
			attributes.add(attr);
			attrIndex++;
			updatedLabelIndices[i] = attrIndex;
		}

		template = new Instances("templateMedoid", attributes, 0);
	}

	/**
	 * Determines the number of cluster depending on the values of the properties
	 * percentage and numClusters.
	 * 
	 * @throws Exception To be handled in an upper level.
	 */
	void configureClusterer() throws Exception {

		int value = 0;

		System.out.println("Clusterer configuration");
		if (percentage == -1 && numClusters == -1) {
			/*
			 * If the number of clusters is not set neither by percentage nor by the
			 * numClusters property, it will be considered by default a 20% of the training
			 * bags.
			 */
			value = (int) (dataset.getNumBags() * 0.2);
			System.out.println("\tnumClusters is 20% of train bags (default configuration) -> " + value + " clusters");
		} else {
			if (percentage != -1) {

				value = (int) (dataset.getNumBags() * percentage);
				System.out.println("\tnumClusters is " + Math.round(percentage * 100) + "% of train bags -> " + value
						+ " clusters");
			} else {
				if (numClusters != -1) {
					value = numClusters;
					System.out.println("\tnumClusters is: " + numClusters);
				}
			}
		}

		// Sets number of clusters in both the transformer and in the clusterer.
		setNumClusters(value);

		// Sets the seed in both the transformer and in the clusterer.
		System.out.println("\tSeed for clustering: " + seed);
		setSeed(seed);

	}

	// --------------------
	// GETTERS AND SETTERS
	// --------------------

	/**
	 * Returns the number of clusters.
	 * 
	 * @return Returns the number of clusters to perform clustering.
	 * @throws Exception To be handled in an upper level.
	 */
	public int getNumClusters() throws Exception {
		return numClusters;
	}

	/**
	 * Sets the number of clusters to perform clustering in both the transformer and
	 * in the clusterer. If the clusterer is null the value of the property is only
	 * set in the transformer and the transformDataset method will establish this
	 * numClusters value in the clusterer after creating it.
	 * 
	 * @param numClusters A number of clusters.
	 * @throws Exception To be handled in an upper level.
	 */
	public void setNumClusters(int numClusters) throws Exception {
		if (clusterer != null)
			clusterer.setNumClusters(numClusters);
		this.numClusters = numClusters;
	}

	/**
	 * Gets the value of the percentage property.
	 * 
	 * @return The percentage of the train instances used as
	 */
	public double getPercentage() {
		return percentage;
	}

	/**
	 * Sets the value of the percentage property.
	 * 
	 * @param percentage The percentage value in [0, 1], for instance 0.2 means that
	 *                   the number of clusters is 20% the number of bags.
	 */
	public void setPercentage(double percentage) {
		this.percentage = percentage;
	}

	/**
	 * Gets the maximum number of iterations used by the clusterer.
	 * 
	 * @return The maximum number of iterations.
	 */
	public int getMaxIterations() {
		return this.clusterer.getMaxIterations();
	}

	/**
	 * Sets the maximum number of iterations for clustering. This method must be
	 * called before clustering.
	 * 
	 * @param maxIterations The maximum number of iterations for clustering.
	 */
	public void setMaxIterations(int maxIterations) {
		this.clusterer.setMaxIterations(maxIterations);
	}

	/**
	 * Returns the distance function used for clustering.
	 * 
	 * @return The distance function used for clustering.
	 */
	public IDistance getDistanceFunction() {
		return this.clusterer.getDistanceFunction();
	}

	/**
	 * Sets the distance function to use for clustering. This method must be called
	 * before clustering.
	 * 
	 * @param distanceFunction The distance function used for clustering.
	 */
	public void setDistanceFunction(IDistance distanceFunction) {
		this.clusterer.setDistanceFunction(distanceFunction);
	}

	/**
	 * Sets the value of the seed used for clustering in both the transformer and in
	 * the clusterer. If the clusterer is null the value of the property is only set
	 * in the transformer and the transformDataset method will establish this seed
	 * value in the clusterer after creating it.
	 * 
	 * @param seed The seed
	 */
	public void setSeed(int seed) {

		if (this.clusterer != null) {
			clusterer.setSeed(seed);
		}

		this.seed = seed;
	}

}
