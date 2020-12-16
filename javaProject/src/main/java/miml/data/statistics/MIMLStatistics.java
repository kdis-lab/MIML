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

import java.util.HashMap;
import java.util.Set;

import miml.data.MIMLInstances;
import mulan.data.LabelSet;

/**
 * Class with methods to obtain information about a MIML dataset.
 * 
 * This java class is based on MLStatistic and MILStatistic.
 * 
 * @author Amelia Zafra
 * @author Eva Gibaja
 * @version 2018/06/15
 */
public class MIMLStatistics {

	/** A MIML data set */
	MIMLInstances dataSet;

	/**
	 * Class with methods to obtain information about a MI dataset.
	 * 
	 * @see MIStatistics
	 */
	protected MIStatistics milstatistics;
	/**
	 * Class with methods to obtain information about a ML dataset.
	 * 
	 * @see MLStatistics
	 */
	protected MLStatistics mlstatistics;

	/**
	 * Constructor.
	 * 
	 * @param dataSet A MIML data set.
	 */
	public MIMLStatistics(MIMLInstances dataSet) {
		this.dataSet = dataSet;
		mlstatistics = new MLStatistics(dataSet.getMLDataSet());
		milstatistics = new MIStatistics(dataSet.getDataSet());
		mlstatistics.calculateStats();
		milstatistics.calculateStats();
	}

	/**
	 * Gets the Phi correlation matrix. It requires the method calculatePhiChi2 to
	 * be previously called.
	 * 
	 * @return phi.
	 */
	public double[][] getPhi() {
		return mlstatistics.getPhi();
	}

	/**
	 * Gets the Chi2 correlation matrix. It requires the method calculatePhiChi2 to
	 * be previously called.
	 * 
	 * @return chi2.
	 */
	public double[][] getChi2() {
		return mlstatistics.getChi2();
	}

	/**
	 * Computes the Cardinality as the average number of labels per pattern. It
	 * requires the method calculateStats to be previously called.
	 * 
	 * @return double
	 */
	public double cardinality() {
		return mlstatistics.cardinality();
	}

	/**
	 * Computes the density as the cardinality/numLabels. It the method
	 * calculateStats to be previously called.
	 * 
	 * @return density.
	 */
	public double density() {
		return mlstatistics.density();
	}

	/**
	 * Returns the prior probabilities of the labels. It requires the method
	 * calculateStats to be previously called.
	 * 
	 * @return An array of double with prior probabilities of labels.
	 */
	public double[] priors() {
		return mlstatistics.priors();
	}

	/**
	 * Returns a set with the distinct label sets of the dataset. It requires the
	 * method calculateStats to be previously called.
	 * 
	 * @return Set of distinct label sets.
	 */
	public Set<LabelSet> labelSets() {
		return mlstatistics.labelSets();
	}

	/**
	 * Returns the frequency of a label set in the dataset. It requires the method
	 * calculateStats to be previously called.
	 * 
	 * @param x A labelset.
	 * @return The frequency of the given labelset.
	 */
	public int labelSetFrequency(LabelSet x) {
		return mlstatistics.labelSetFrequency(x);
	}

	/**
	 * Returns the HashMap containing the distinct labelsets and their frequencies.
	 * It requires the method calculateStats to be previously called.
	 * 
	 * @return HashMap with distinct labelsest and their frequencies.
	 */
	public HashMap<LabelSet, Integer> labelCombCount() {
		return mlstatistics.labelCombCount();
	}

	/**
	 * This method calculates a matrix with the coocurrences of pairs of labels. It
	 * requires the method calculateStats to be previously called.
	 *
	 * @param mlDataSet A multi-label dataset.
	 * @return A coocurrences matrix of pairs of labels.
	 */
	public double[][] calculateCooncurrence(MIMLInstances mlDataSet) {
		return mlstatistics.calculateCoocurrence(mlDataSet.getMLDataSet());
	}

	/**
	 * Calculates Phi and Chi-square correlation matrix.
	 *
	 * @param dataSet A multi-label dataset.
	 * @throws java.lang.Exception To be handled in an upper level.
	 */
	public void calculatePhiChi2(MIMLInstances dataSet) throws Exception {

		mlstatistics.calculatePhiChi2(dataSet.getMLDataSet());
	}

	/**
	 * Calculates a histogram of Phi correlations. It requires the method
	 * calculatePhi to be previously called.
	 *
	 * @return An array with Phi correlations.
	 */
	public double[] getPhiHistogram() {
		return mlstatistics.getPhiHistogram();
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
		return mlstatistics.uncorrelatedLabels(labelIndex, bound);
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
		return mlstatistics.topPhiCorrelatedLabels(labelIndex, k);
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
		mlstatistics.printPhiDiagram(step);
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
		return mlstatistics.innerClassIR();
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
		return mlstatistics.innerClassIR();
	}

	/**
	 * Computes the average of any IR vector.
	 * 
	 * @param IR An IR vector previously computed
	 * @return double
	 */
	public double averageIR(double[] IR) {
		return mlstatistics.averageIR(IR);
	}

	/**
	 * Computes the variance of any IR vector.
	 * 
	 * @param IR An IR vector previously computed.
	 * @return Variance of any IR vector.
	 */
	public double varianceIR(double[] IR) {
		return mlstatistics.varianceIR(IR);
	}

	/**
	 * Returns proportion of unique label combinations (pPunique) value defined as
	 * the proportion of labelsets which are unique across the total number of
	 * examples. It requires the method calculateStats to be previously called.
	 * 
	 * More information in Jesse Read. 2010. Scalable Multi-label Classification.
	 * Ph.D. Dissertation. University of Waikato.
	 * 
	 * @return Proportion of unique label combinations.
	 */
	public double pUnique() {
		return mlstatistics.pUnique();
	}

	/**
	 * Returns pMax, the proportion of examples associated with the most frequently
	 * occurring labelset. It requires the method calculateStats to be previously
	 * called.
	 * 
	 * More information in Jesse Read. 2010. Scalable Multi-label Classification.
	 * Ph.D. Dissertation. University of Waikato.
	 * 
	 * @return pMax.
	 */
	public double pMax() {
		// return (1.0*maxCount*peak)/numInstances; to consider two or more most
		// frequent labelSets
		return mlstatistics.pMax();
	}

	/**
	 * Computes the IR for each labelSet as (patterns of majorityLabelSet)/(patterns
	 * of the labelSet). It requires the method calculateStats to be previously
	 * called.
	 * 
	 * @return HashMap&lt;LabelSet, Double&gt;
	 */
	public HashMap<LabelSet, Double> labelSkew() {
		return mlstatistics.labelSkew();
	}

	/**
	 * Computes the average labelSkew.
	 * 
	 * @param skew The IR for each labelSet previously computed.
	 * @return Average labelSkew.
	 */
	public double averageSkew(HashMap<LabelSet, Double> skew) {
		return mlstatistics.averageSkew(skew);
	}

	/**
	 * Computes the skewRatio as peak/base. It requires the method calculateStats to
	 * be previously called.
	 * 
	 * @return SkewRatio as peak/base.
	 */
	public double skewRatio() {
		return mlstatistics.skewRatio();
	}

	/**
	 * Returns statistics in textual representation. It requires the method
	 * calculateStats to be previously called.
	 * 
	 * @return Statistics in textual representation.
	 * 
	 * 
	 */
	// TODO debe revisarse este método
	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();

		sb.append(milstatistics.toString());

		sb.append(mlstatistics.toString());

		return sb.toString();
	}

	/**
	 * Returns statistics in CSV representation. It requires the method
	 * calculateStats to be previously called.
	 * 
	 * @return Statistics in CSV representation.
	 */
	public String toCSV() {
		StringBuffer sb = new StringBuffer();
		return sb.toString();
	}

	/**
	 * Returns labelSkew in textual representation.
	 * 
	 * @param skew The IR for each labelSet previously computed.
	 * @return LabelSkew in textual representation.
	 */
	protected String distributionBagsToString(HashMap<LabelSet, Double> skew) {

		return (mlstatistics.distributionBagsToString(skew));
	}

	/**
	 * Returns labelSkew in CSV representation.
	 * 
	 * @param skew The IR for each labelSet previously computed.
	 * @return LabelSkew in CSV representation.
	 */
	protected String distributionBagsToCSV(HashMap<LabelSet, Double> skew) {

		return (mlstatistics.distributionBagsToCSV(skew));
	}

	/**
	 * Returns cooCurrenceMatrix in textual representation. It requires the method
	 * calculateCooncurrence to be previously called.
	 * 
	 * @return CooCurrenceMatrix in textual representation.
	 */
	public String cooncurrenceToString() {

		return mlstatistics.coocurrenceToString();
	}

	/**
	 * Returns cooCurrenceMatrix in CSV representation. It requires the method
	 * calculateCooncurrence to be previously called.
	 * 
	 * @return CooCurrenceMatrix in CSV representation.
	 */
	public String cooncurrenceToCSV() {

		return mlstatistics.coocurrenceToCSV();
	}

	/**
	 * Returns Phi correlations in textual representation. It requires the method
	 * calculatePhiChi2 to be previously called.
	 * 
	 * @param matrix Matrix with Phi correlations.
	 * 
	 * @return Phi correlations in textual representation.
	 */
	public String correlationsToString(double matrix[][]) {

		return mlstatistics.correlationsToString(matrix);
	}

	/**
	 * Returns Phi correlations in CSV representation. It requires the method
	 * calculatePhiChi2 to be previously called.
	 * 
	 * @param matrix Matrix with Phi correlations.
	 * @return Phi correlations in CSV representation.
	 */
	public String correlationsToCSV(double matrix[][]) {

		return mlstatistics.correlationsToCSV(matrix);
	}

	/**
	 * Returns distributionBags in textual representation.
	 * 
	 * @return String with bags distribution.
	 */
	protected String distributionBagsToString() {

		return (milstatistics.distributionBagsToString());
	}

	/**
	 * Returns distributionBags in CSV representation.
	 * 
	 * @return CSV with bags distribution.
	 */
	protected String distributionBagsToCSV() {

		return (milstatistics.distributionBagsToCSV());
	}

	/**
	 * Returns the dataset used to calculate the statistics.
	 * 
	 * @return  A MIML data set.
	 */
	public MIMLInstances getDataSet() {
		return dataSet;
	}

	/**
	 * Set the dataset used.
	 * 
	 * @param dataSet  A MIML data set.
	 */
	public void setDataSet(MIMLInstances dataSet) {
		this.dataSet = dataSet;
	}
}
