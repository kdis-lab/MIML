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

package miml.data.statistics;

import java.util.HashMap;

import weka.core.Instances;

/**
 * Class with methods to obtain information about a MI dataset such as the
 * number of attributes per bag, the average number of instances per bag, and
 * the distribution of number of instances per bag...
 * 
 * @author F.J. Gonzalez
 * @author Eva Gigaja
 * @version 20150925
 */
public class MIStatistics {
	/** The minimum number of instances per bag. */
	int minInstancesPerBag;
	/** The maximum number of instances per bag. */
	int maxInstancesPerBag;
	/** The average number of instances per bag. */
	double avgInstancesPerBag;
	/** The number of attributes per bag. */
	int attributesPerBag;
	/** The number of bags. */
	int numBags;
	/** The total of instances. */
	int totalInstances;
	/** The distribution of number of instances per bag. */
	HashMap<Integer, Integer> distributionBags;
	/** Instances dataset*/
	Instances dataSet;
	
	public MIStatistics(Instances dataSet) {
		this.dataSet = dataSet;
		calculateStats();
	}
	
	/**
	 * Calculates various MIML statistics, such as instancesPerBag and
	 * attributesPerBag.
	 * 
	 */
	protected void calculateStats() {
		numBags = dataSet.numInstances();
		attributesPerBag = dataSet.instance(0).relationalValue(1).numAttributes();
		minInstancesPerBag = Integer.MAX_VALUE;
		maxInstancesPerBag = Integer.MIN_VALUE;
		totalInstances = 0;

		// Each pair <Integer, Integer> stores <numberOfInstances, numberOfBags>
		distributionBags = new HashMap<Integer, Integer>();
		for (int i = 0; i < numBags; i++) {
			int nInstances = dataSet.instance(i).relationalValue(1).numInstances();
			totalInstances += nInstances;

			if (nInstances < minInstancesPerBag) {
				minInstancesPerBag = nInstances;
			}
			if (nInstances > maxInstancesPerBag) {
				maxInstancesPerBag = nInstances;
			}
			if (distributionBags.containsKey(nInstances)) {
				distributionBags.put(nInstances, distributionBags.get(nInstances) + 1);
			} else {
				distributionBags.put(nInstances, 1);
			}
		}

		avgInstancesPerBag = 0.0;
		for (Integer set : distributionBags.keySet()) {
			avgInstancesPerBag += set * distributionBags.get(set);
		}
		avgInstancesPerBag = avgInstancesPerBag / numBags;
	}

	/**
	 * Returns distributionBags in textual representation.
	 * 
	 * @return DistributionBags in textual representation.
	 */
	protected String distributionBagsToString() {
		StringBuilder sb = new StringBuilder();
		for (Integer set : distributionBags.keySet()) {
			sb.append("\n\t<" + distributionBags.get(set) + "," + set + ">");
		}
		return (sb.toString());
	}

	/**
	 * Returns distributionBags in CSV representation.
	 * 
	 * @return DistributionBags in CSV representation.
	 */
	protected String distributionBagsToCSV() {
		StringBuilder sb = new StringBuilder();
		for (Integer set : distributionBags.keySet()) {
			sb.append("\n" + distributionBags.get(set) + ";" + set);
		}
		return (sb.toString());
	}

	/**
	 * Returns statistics in textual representation.
	 * 
	 * @return Statistics in textual representation.
	 */
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("\n----------------------------");
		sb.append("\nMIL Statistics--------------");
		sb.append("\n----------------------------");
		sb.append("\nnBags: " + numBags);
		sb.append("\nTotalInstances: " + totalInstances);
		sb.append("\nAvgInstancesPerBag: " + avgInstancesPerBag);
		sb.append("\nMinInstancesPerBag: " + minInstancesPerBag);
		sb.append("\nMaxInstancesPerBag: " + maxInstancesPerBag);
		sb.append("\nAttributesPerBag: " + attributesPerBag);
		sb.append("\nDistribution of bags <nBags, nInstances>:");
		sb.append(distributionBagsToString());
		return (sb.toString());
	}

	/**
	 * Returns statistics in CSV representation.
	 * 
	 * @return Statistics in CSV representation.
	 */
	public String toCSV() {
		StringBuilder sb = new StringBuilder();
		sb.append("\nMIL STATISTICS:");
		sb.append("\nnBags;" + numBags);
		sb.append("\nTotalInstances: " + totalInstances);
		sb.append("\nAvgInstancesPerBag;" + avgInstancesPerBag);
		sb.append("\nMinInstancesPerBag;" + minInstancesPerBag);
		sb.append("\nMaxInstancesPerBag;" + maxInstancesPerBag);
		sb.append("\nAttributesPerBag;" + attributesPerBag);
		sb.append("\nDistribution of bags <nBags, nInstances>");
		sb.append(distributionBagsToCSV());
		return (sb.toString());
	}

}
