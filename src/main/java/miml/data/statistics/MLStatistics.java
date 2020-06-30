/*
 *    This program is free software; you can redistribute it and/or modify
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

/*
 *    MLStatistics.java
 *    This java class is based on the mulan.data.Statistics.java 
 *    class provided in the mulan java framework for multi-label learning
 *    Tsoumakas, G., Katakis, I., Vlahavas, I. (2010) "Mining Multi-label Data", 
 *    Data Mining and Knowledge Discovery Handbook, O. Maimon, L. Rokach (Ed.),
 *    Springer, 2nd edition, 2010.
 */

package miml.data.statistics;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;

import mulan.data.LabelSet;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Class with methods to obtain information about a ML dataset.
 * 
 * This java class is based on the mulan.data.Statistics.java class provided in
 * the Mulan java framework for multi-label learning Tsoumakas, G., Katakis, I.,
 * Vlahavas, I. (2010) "Mining Multi-label Data", Data Mining and Knowledge
 * Discovery Handbook, O. Maimon, L. Rokach (Ed.), Springer, 2nd edition, 2010.
 * Our contribution is mainly related with methods to measure the degree of
 * imbalance and a fixed bug in the method printPhiDiagram.
 * 
 * @author F.J. Gonzalez
 * @author Eva Gibaja
 * @version 20150925
 */
public class MLStatistics {
	
	// Basic features
	/** The number of labels. */
	protected int numLabels;
	/** The number of examples. */
	protected int numExamples;
	/** The number of attributes. */
	protected int numAttributes;
	/** The number of nominal predictive attributes. */
	protected int numNominal = 0;
	/** The number of numeric attributes. */
	protected int numNumeric = 0;

	// Label-based features
	/** The number of positive examples per label. */
	protected int[] positiveExamplesPerLabel;
	/** The number of examples having 0, 1, 2,... , numLabel labels. */
	protected int[] distributionLabelsPerExample;

	// LabelSet-based features
	/** LabelSets in the dataset. */
	protected HashMap<LabelSet, Integer> labelCombinations;
	/** The highest labelSet count. */
	protected int peak;
	/** The lowest labelSet count. */
	protected int base;
	/** Number of labelSets with only one pattern. */
	protected int nUnique;
	/** Number of labelSets with the peak value. */
	protected int maxCount;

	// features not pre-computed
	/** Coocurrence matrix. */
	double coocurrenceMatrix[][];
	/**
	 * Phi matrix values in [-1,1] where -1 = inverse relation, 0 = no relation, 1 =
	 * direct relation.
	 */
	double phi[][] = null;
	/**
	 * Chi square matrix values where 0 = complete independence. Values larger than
	 * 6.63 show label dependence at 0.01 level of significance (99%). Values larger
	 * than 3.84 show label dependence at 0.05 level of significance (95%).
	 */
	double chi2[][] = null;

	/** Multi label dataset */
	private MultiLabelInstances mlDataSet;
	
	/**
	 * Constructor.
	 * 
	 * @param mlDataSet MultiLabel dataset.
	 */
	public MLStatistics(MultiLabelInstances mlDataSet) {
		this.mlDataSet = mlDataSet;
		calculateStats();
	}
	
	/**
	 * Gets the Phi correlation matrix. It requires the method calculatePhiChi2 to
	 * be previously called.
	 * 
	 * @return phi
	 */
	public double[][] getPhi() {
		return phi;
	}

	/**
	 * Gets the Chi2 correlation matrix. It requires the method calculatePhiChi2 to
	 * be previously called.
	 * 
	 * @return chi2
	 */
	public double[][] getChi2() {
		return chi2;
	}

	/**
	 * Calculates various ML statistics.
	 */
	protected void calculateStats() {

		// Initialize basic properties
		numLabels = mlDataSet.getNumLabels();
		numExamples = mlDataSet.getNumInstances();

		// Gather information about attributes
		int[] featureIndices = mlDataSet.getFeatureIndices();
		numAttributes = featureIndices.length;
		for (int i = 0; i < featureIndices.length; i++) {
			if (mlDataSet.getDataSet().attribute(featureIndices[i]).isNominal()) {
				numNominal++;
			}
			if (mlDataSet.getDataSet().attribute(featureIndices[i]).isNumeric()) {
				numNumeric++;
			}
		}

		// Obtain the number of positive examples per label and the different
		// labelSets of the dataset and their count
		labelCombinations = new HashMap<LabelSet, Integer>();
		positiveExamplesPerLabel = new int[numLabels];

		distributionLabelsPerExample = new int[numLabels + 1]; // 0..numLabels
		int[] labelIndices = mlDataSet.getLabelIndices();
		for (int i = 0; i < numExamples; i++) {
			int labelCount = 0; // number of labels of the current example
			double labelComb[] = new double[numLabels]; // labelset of the
														// current
														// label
			for (int j = 0; j < numLabels; j++) {
				if (mlDataSet.getDataSet().instance(i).stringValue(labelIndices[j]).equals("1")) {
					positiveExamplesPerLabel[j]++;
					labelCount++;
					labelComb[j] = 1;
				}
			}
			distributionLabelsPerExample[labelCount]++;
			// Update the labelSet
			LabelSet labelSet = new LabelSet(labelComb);
			if (labelCombinations.containsKey(labelSet)) {
				labelCombinations.put(labelSet, labelCombinations.get(labelSet) + 1);
			} else {
				labelCombinations.put(labelSet, 1);
			}
		}

		// Obtain the peak and base values
		peak = Integer.MIN_VALUE;
		base = Integer.MAX_VALUE;
		nUnique = 0;
		maxCount = 0;
		for (LabelSet set : labelCombinations.keySet()) {
			Integer value = labelCombinations.get(set);
			if (value > peak) {
				peak = value;
				maxCount = 0;
			}
			if (value < base)
				base = value;
			if (value == 1)
				nUnique++;
			if (value == peak)
				maxCount++;
		}
		return;
	}

	/**
	 * Computes the Cardinality as the average number of labels per pattern. It
	 * requires the method calculateStats to be previously called.
	 * 
	 * @return double
	 */
	public double cardinality() {
		double sum = 0;
		for (int i = 0; i < positiveExamplesPerLabel.length; i++)
			sum += positiveExamplesPerLabel[i];
		return sum / numExamples;
	}

	/**
	 * Computes the density as the cardinality/numLabels. It the method
	 * calculateStats to be previously called.
	 * 
	 * @return double
	 */
	public double density() {
		double sum = 0;
		for (int i = 0; i < positiveExamplesPerLabel.length; i++)
			sum += positiveExamplesPerLabel[i];
		return (sum / numExamples) / numLabels;
	}

	/**
	 * Returns the prior probabilities of the labels. It requires the method
	 * calculateStats to be previously called.
	 * 
	 * @return An array of double with prior probabilities of labels.
	 */
	public double[] priors() {
		double probabilities[] = new double[numLabels];
		for (int i = 0; i < probabilities.length; i++)
			probabilities[i] = positiveExamplesPerLabel[i] / numExamples;
		return probabilities;
	}

	/**
	 * Returns a set with the distinct label sets of the dataset. It requires the
	 * method calculateStats to be previously called.
	 * 
	 * @return Set of distinct label sets.
	 */
	public Set<LabelSet> labelSets() {
		return labelCombinations.keySet();
	}

	/**
	 * Returns the frequency of a label set in the dataset. It requires the method
	 * calculateStats to be previously called.
	 * 
	 * @param x A labelset.
	 * @return The frequency of the given labelset.
	 */
	public int labelSetFrequency(LabelSet x) {
		return labelCombinations.get(x);
	}

	/**
	 * Returns the HashMap containing the distinct labelsets and their frequencies.
	 * It requires the method calculateStats to be previously called.
	 * 
	 * @return HashMap with distinct labelsest and their frequencies.
	 */
	public HashMap<LabelSet, Integer> labelCombCount() {
		return labelCombinations;
	}

	/**
	 * This method calculates a matrix with the coocurrences of pairs of labels. It
	 * requires the method calculateStats to be previously called.
	 *
	 * @param mlDataSet A multi-label dataset.
	 * @return A coocurrences matrix of pairs of labels.
	 */
	public double[][] calculateCoocurrence(MultiLabelInstances mlDataSet) {
		coocurrenceMatrix = new double[numLabels][numLabels];
		int[] labelIndices = mlDataSet.getLabelIndices();
		for (int k = 0; k < numExamples; k++) {
			Instance temp = mlDataSet.getDataSet().instance(k);
			for (int i = 0; i < numLabels; i++) {
				for (int j = 0; j < numLabels; j++) {
					if ((i < j) && (temp.stringValue(labelIndices[i]).equals("1")
							&& temp.stringValue(labelIndices[j]).equals("1"))) {
						coocurrenceMatrix[i][j]++;
					}
				}
			}
		}
		return coocurrenceMatrix;
	}

	/**
	 * Calculates Phi and Chi-square correlation matrix.
	 *
	 * @param dataSet A multi-label dataset.
	 * @throws java.lang.Exception To be handled in an upper level.
	 */
	public void calculatePhiChi2(MultiLabelInstances dataSet) throws Exception {
		numLabels = dataSet.getNumLabels();

		// The indices of the label attributes
		int[] labelIndices;

		labelIndices = dataSet.getLabelIndices();
		numLabels = dataSet.getNumLabels();
		phi = new double[numLabels][numLabels];
		chi2 = new double[numLabels][numLabels];

		Remove remove = new Remove();
		remove.setInvertSelection(true);
		remove.setAttributeIndicesArray(labelIndices);
		remove.setInputFormat(dataSet.getDataSet());
		Instances result = Filter.useFilter(dataSet.getDataSet(), remove);
		result.setClassIndex(result.numAttributes() - 1);

		for (int i = 0; i < numLabels; i++) {
			int a[] = new int[numLabels];
			int b[] = new int[numLabels];
			int c[] = new int[numLabels];
			int d[] = new int[numLabels];
			double e[] = new double[numLabels];
			double f[] = new double[numLabels];
			double g[] = new double[numLabels];
			double h[] = new double[numLabels];
			for (int j = 0; j < result.numInstances(); j++) {
				for (int l = 0; l < numLabels; l++) {
					if (result.instance(j).stringValue(i).equals("0")) {
						if (result.instance(j).stringValue(l).equals("0")) {
							a[l]++;
						} else {
							c[l]++;
						}
					} else {
						if (result.instance(j).stringValue(l).equals("0")) {
							b[l]++;
						} else {
							d[l]++;
						}
					}
				}
			}
			for (int l = 0; l < numLabels; l++) {
				e[l] = a[l] + b[l];
				f[l] = c[l] + d[l];
				g[l] = a[l] + c[l];
				h[l] = b[l] + d[l];
				double mult = e[l] * f[l] * g[l] * h[l];
				double denominator = Math.sqrt(mult);
				double nominator = a[l] * d[l] - b[l] * c[l];
				phi[i][l] = nominator / denominator;
				chi2[i][l] = phi[i][l] * phi[i][l] * (a[l] + b[l] + c[l] + d[l]);
			}
		}
	}

	/**
	 * Calculates a histogram of Phi correlations. It requires the method
	 * calculatePhi to be previously called.
	 *
	 * @return An array with Phi correlations.
	 */
	public double[] getPhiHistogram() {
		double[] pairs = new double[numLabels * (numLabels - 1) / 2];
		int counter = 0;
		for (int i = 0; i < numLabels - 1; i++) {
			for (int j = i + 1; j < numLabels; j++) {
				pairs[counter] = phi[i][j];
				counter++;
			}
		}
		return pairs;
	}

	/**
	 * Returns the indices of the labels whose Phi coefficient values lie between
	 * -bound &lt;= phi &lt;= bound. It requires the method calculatePhi to be
	 * previously called.
	 *
	 * @param labelIndex The label index.
	 * @param bound      The bound.
	 * @return The indices of the labels whose Phi coefficient values lie between
	 *         -bound &lt;= phi &lt;= bound.
	 */
	public int[] uncorrelatedLabels(int labelIndex, double bound) {
		ArrayList<Integer> indiceslist = new ArrayList<Integer>();
		for (int i = 0; i < numLabels; i++) {
			if (Math.abs(phi[labelIndex][i]) <= bound) {
				indiceslist.add(i);
			}
		}
		int[] indices = new int[indiceslist.size()];
		for (int i = 0; i < indiceslist.size(); i++) {
			indices[i] = indiceslist.get(i);
		}
		return indices;
	}

	/**
	 * Returns the indices of the labels that have the strongest Phi correlation
	 * with the label which is given as a parameter. The second parameter is the
	 * number of labels that will be returned. It requires the method calculatePhi
	 * to be previously called.
	 *
	 * @param labelIndex The label index.
	 * @param k          The number of labels that will be returned. The number of
	 *                   labels that will be returned.
	 * @return The indices of the k most correlated labels.
	 */
	public int[] topPhiCorrelatedLabels(int labelIndex, int k) {
		// create a new array containing the absolute values of the original
		// array
		double[] absCorrelations = new double[numLabels];
		for (int i = 0; i < numLabels; i++) {
			absCorrelations[i] = Math.abs(phi[labelIndex][i]);
		}
		// sort the array of correlations
		int[] sorted = Utils.stableSort(absCorrelations);

		int[] topPhiCorrelated = new int[k + 1];
		// the k last values of the sorted array are the indices of the top k
		// correlated labels
		for (int i = 0; i < k; i++) {
			topPhiCorrelated[i] = sorted[numLabels - 1 - i];
		}
		// one more for the class
		topPhiCorrelated[k] = numLabels;

		return topPhiCorrelated;
	}

	/**
	 * This method prints data, useful for the visualization of Phi per dataset. It
	 * prints int(1/step) + 1 pairs of values. The first value of each pair is the
	 * phi value and the second is the average number of labels that correlate to
	 * the rest of the labels with correlation higher than the specified Phi value.
	 * It requires the method calculatePhi to be previously called.
	 *
	 * @param step The Ohi value increment step.
	 */
	public void printPhiDiagram(double step) {
		String pattern = "0.00";
		DecimalFormat myFormatter = new DecimalFormat(pattern);
		System.out.println("Phi      AvgCorrelated");
		double tempPhi = 0;
		while (tempPhi <= 1.001) {
			double avgCorrelated = 0;
			for (int i = 0; i < numLabels; i++) {
				int[] temp = uncorrelatedLabels(i, tempPhi);
				avgCorrelated += (numLabels - temp.length);
			}
			avgCorrelated /= numLabels;
			// Bug fixed in the current project
			// System.out.println(myFormatter.format(phi) + " " +
			// avgCorrelated);
			System.out.println(myFormatter.format(tempPhi) + "     " + avgCorrelated);
			tempPhi += step;
		}
	}

	/**
	 * Computes the innerClassIR for each label as
	 * negativePatterns/positivePatterns. It requires the method calculateStats to
	 * be previously called.
	 * 
	 * @return An IR for each label: negativePatterns/positivePatterns.
	 */
	public double[] innerClassIR() { // IR for each label:
										// negativePatterns/positivePatterns
		double IR[] = new double[numLabels];
		for (int i = 0; i < numLabels; i++) {
			IR[i] = (numExamples - positiveExamplesPerLabel[i]) / positiveExamplesPerLabel[i];
		}
		return IR;
	}

	/**
	 * Computes the interClassIR for each label
	 * positiveExamplesOfMajorityLabel/positivePatternsLabel. It requires the method
	 * calculateStats to be previously called.
	 * 
	 * @return An IR between binary labels:
	 *         maxPositiveClassExamples/positiveExamplesLabel.
	 */
	public double[] interClassIR() { // IR between binary labels:
										// maxPositiveClassExamples/positiveExamplesLabel
		double IR[] = new double[numLabels];
		int max = Utils.maxIndex(positiveExamplesPerLabel);
		for (int i = 0; i < numLabels; i++) {
			IR[i] = positiveExamplesPerLabel[max] / positiveExamplesPerLabel[i];
		}
		return IR;
	}

	/**
	 * Computes the average of any IR vector.
	 * 
	 * @param IR An IR vector previously computed
	 * @return double
	 */
	public double averageIR(double[] IR) {
		return Utils.mean(IR);
	}

	/**
	 * Computes the variance of any IR vector.
	 * 
	 * @param IR An IR vector previously computed.
	 * @return double.
	 */
	public double varianceIR(double[] IR) {
		return Utils.variance(IR);
	}

	/**
	 * Returns proportion of unique label combinations (pPunique) value defined as
	 * the proportion of labelsets which are unique across the total number of
	 * examples. It requires the method calculateStats to be previously called.
	 * 
	 * More information in Jesse Read. 2010. Scalable Multi-label Classification.
	 * Ph.D. Dissertation. University of Waikato.
	 * 
	 * @return double
	 */
	public double pUnique() {
		return (1.0 * nUnique) / numExamples;
	}

	/**
	 * Returns pMax, the proportion of examples associated with the most frequently
	 * occurring labelset. It requires the method calculateStats to be previously
	 * called.
	 * 
	 * More information in Jesse Read. 2010. Scalable Multi-label Classification.
	 * Ph.D. Dissertation. University of Waikato.
	 * 
	 * @return double
	 */
	public double pMax() {
		// return (1.0*maxCount*peak)/numInstances; to consider two or more most
		// frequent labelSets
		return (1.0 * peak) / numExamples;
	}

	/**
	 * Computes the IR for each labelSet as (patterns of majorityLabelSet)/(patterns
	 * of the labelSet). It requires the method calculateStats to be previously
	 * called.
	 * 
	 * @return HashMap&lt;LabelSet, Double&gt;
	 */
	public HashMap<LabelSet, Double> labelSkew() {
		// IR between labelSets
		HashMap<LabelSet, Double> IR = new HashMap<LabelSet, Double>();
		for (LabelSet set : labelCombinations.keySet()) {
			Integer value = labelCombinations.get(set);
			IR.put(set, (peak * 1.0) / value);
		}
		return IR;
	}

	/**
	 * Computes the average labelSkew.
	 * 
	 * @param skew The IR for each labelSet previously computed.
	 * @return double
	 */
	public double averageSkew(HashMap<LabelSet, Double> skew) {
		double result = 0.0;
		for (LabelSet set : skew.keySet()) {
			result += skew.get(set);
		}
		return (result / skew.size());
	}

	/**
	 * Computes the skewRatio as peak/base. It requires the method calculateStats to
	 * be previously called.
	 * 
	 * @return double
	 */
	public double skewRatio() {
		return (1.0 * peak) / base;
	}

	/**
	 * Returns statistics in textual representation. It requires the method
	 * calculateStats to be previously called.
	 * 
	 * @return string
	 */
	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append("\n-------------------------");
		sb.append("\n ML - Summary------------------");
		sb.append("\n-------------------------");
		sb.append("\nNumber of labels: " + numLabels);
		sb.append("\nNumber of examples: " + numExamples);
		sb.append("\nNumber of attributes (without labels):" + numAttributes);
		sb.append("\nNumber of attributes (including labels):" + (numAttributes + numLabels));
		sb.append("\n\tnumeric: " + numNumeric);
		sb.append("\n\tnominal:" + numNominal);
		sb.append("\nCardinality:" + cardinality());
		sb.append("\nDensity:" + density());
		sb.append("\nExamples of cardinality:");
		for (int j = 0; j <= numLabels; j++) {
			sb.append("\n\t[" + j + "]:" + this.distributionLabelsPerExample[j]);
		}

		sb.append("\n-------------------------");
		sb.append("\nLabel-based statistics---");
		sb.append("\n-------------------------");
		sb.append("\nPositive examples per label: ");
		for (int i = 0; i < positiveExamplesPerLabel.length; i++) {
			sb.append("\n\tLabel[" + i + "]: " + positiveExamplesPerLabel[i]);
		}

		sb.append("\nInnerClassIR:");
		double IR[] = innerClassIR();
		for (int i = 0; i < IR.length; i++) {
			sb.append("\n\tLabel[" + i + "]: " + IR[i]);
		}
		sb.append("\n\tMean: " + averageIR(IR));
		sb.append("\n\tVariance: " + varianceIR(IR));

		sb.append("\nIterClassIR:");
		IR = this.interClassIR();
		for (int i = 0; i < IR.length; i++) {
			sb.append("\n\tLabel[" + i + "]: " + IR[i]);
		}
		sb.append("\n\tMean: " + averageIR(IR));
		sb.append("\n\tVariance: " + varianceIR(IR));

		sb.append("\n----------------------------");
		sb.append("\nLabelSet-based statistics---");
		sb.append("\n----------------------------");
		sb.append("\nDistinct Labelsets:" + labelCombinations.size());
		HashMap<LabelSet, Integer> labelsets = labelCombCount();
		sb.append("\nDistribution of labelSets <LabelSet, Count>:");
		for (LabelSet set : labelsets.keySet()) {
			sb.append("\n\t<" + set + ">, " + labelsets.get(set));
		}
		sb.append("\nPeak: " + peak);
		sb.append("\nBase: " + base);
		sb.append("\nSkewRatio = Peak/Base: " + skewRatio());
		sb.append("\npUnique: " + pUnique());
		sb.append("\npMax: " + pMax());
		sb.append("\nLabelSkew:");
		HashMap<LabelSet, Double> skew = labelSkew();
		sb.append(distributionBagsToString(skew));
		sb.append("\n\tMean: " + averageSkew(skew));

		return sb.toString();
	}

	/**
	 * Returns statistics in CSV representation. It requires the method
	 * calculateStats to be previously called.
	 * 
	 * @return string
	 */
	public String toCSV() {
		StringBuffer sb = new StringBuffer();
		return sb.toString();
	}

	/**
	 * Returns labelSkew in textual representation.
	 * 
	 * @param skew The IR for each labelSet previously computed.
	 * @return string
	 */
	protected String distributionBagsToString(HashMap<LabelSet, Double> skew) {
		StringBuilder sb = new StringBuilder();
		for (LabelSet set : skew.keySet()) {
			sb.append("\n\t<" + set + ">," + skew.get(set));
		}
		return (sb.toString());
	}

	/**
	 * Returns labelSkew in CSV representation.
	 * 
	 * @param skew The IR for each labelSet previously computed.
	 * @return string
	 */
	protected String distributionBagsToCSV(HashMap<LabelSet, Double> skew) {
		StringBuilder sb = new StringBuilder();
		for (LabelSet set : skew.keySet()) {
			sb.append("\n<" + set + ">;" + skew.get(set));
		}
		return (sb.toString());
	}

	/**
	 * Returns coocurrenceMatrix in textual representation. It requires the method
	 * calculateCoocurrence to be previously called.
	 * 
	 * @return string
	 */
	public String coocurrenceToString() {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < numLabels; i++) {
			sb.append("\n");
			for (int j = 0; j < numLabels; j++) {
				sb.append(coocurrenceMatrix[i][j] + "\t");
			}
		}
		return sb.toString();
	}

	/**
	 * Returns coocurrenceMatrix in CSV representation. It requires the method
	 * calculateCoocurrence to be previously called.
	 * 
	 * @return string
	 */
	public String coocurrenceToCSV() {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < numLabels; i++) {
			sb.append("\n");
			for (int j = 0; j < numLabels; j++) {
				sb.append(coocurrenceMatrix[i][j] + ";");
			}
		}
		return sb.toString();
	}

	/**
	 * Returns Phi correlations in textual representation. It requires the method
	 * calculatePhiChi2 to be previously called.
	 * 
	 * @param matrix Matrix with Phi correlations.
	 * 
	 * @return string
	 */
	public String correlationsToString(double matrix[][]) {
		String pattern = "0.00";
		DecimalFormat myFormatter = new DecimalFormat(pattern);
		StringBuilder sb = new StringBuilder("\n");
		for (int i = 0; i < numLabels; i++) {
			for (int j = 0; j < numLabels; j++) {
				if (i != j)
					sb.append(myFormatter.format(matrix[i][j]) + " ");
				else
					sb.append("---");
			}
			sb.append("\n");
		}
		return sb.toString();
	}

	/**
	 * Returns Phi correlations in CSV representation. It requires the method
	 * calculatePhiChi2 to be previously called.
	 * 
	 * @param matrix Matrix with Phi correlations.
	 * @return String
	 */
	public String correlationsToCSV(double matrix[][]) {
		String pattern = "0.00";
		DecimalFormat myFormatter = new DecimalFormat(pattern);
		StringBuilder sb = new StringBuilder("\n");
		for (int i = 0; i < numLabels; i++) {
			for (int j = 0; j < numLabels; j++) {
				if (i != j)
					sb.append(myFormatter.format(matrix[i][j]) + ";");
				else
					sb.append("---;");
			}
			sb.append("\n");
		}
		return sb.toString();
	}
}
