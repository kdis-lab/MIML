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

package miml.tutorial;

import java.io.File;
import miml.clusterers.KMedoids;
import miml.data.MIMLInstances;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 
 * Class to show an example of clustering of a MIML Dataset.
 * 
 * @author Eva Gibaja
 * @version 20230412
 *
 */
public class Clustering {

	public static void main(String[] args) {

		try {
			System.out.println("Loading dataset...");
			MIMLInstances mimlDataSet1 = new MIMLInstances("data" + File.separator + "miml_birds_random_80train.arff",
					"data" + File.separator + "miml_birds.xml");
			MIMLInstances mimlDataSet2 = new MIMLInstances("data" + File.separator + "miml_birds_random_20test.arff",
					"data" + File.separator + "miml_birds.xml");

			KMedoids kmedoids = new KMedoids();
			kmedoids.buildClusterer(mimlDataSet1.getDataSet());
			System.out.println("\nSummary of clustering performed:");
			System.out.println("\tNumber of clusters: " + kmedoids.numberOfClusters());
			System.out.println("\tCost of clustering: " + kmedoids.getConfigurationCost());
			System.out.println("\tNumber of iterations: " + kmedoids.getNumIterations());

			// Checking a train instance
			System.out.println("\n\nChecking a train instance:");
			Instances dataTrain = mimlDataSet1.getDataSet();
			Instance instance = dataTrain.instance(5);

			double[] distances = kmedoids.distanceToMedoids(instance);
			System.out.println("\tDistance to medoids: \n" + weka.core.Utils.arrayToString(distances));

			double[] distribution = kmedoids.distributionForInstance(instance);
			System.out.println("\tDistribution: \n" + weka.core.Utils.arrayToString(distribution));

			System.out.println("\tInstance assigned to Cluster (0, " + (kmedoids.numberOfClusters() - 1) + "): "
					+ kmedoids.clusterInstance(instance));

			// Checking a test instance
			System.out.println("\n\nChecking a test instance:");
			Instances dataTest = mimlDataSet2.getDataSet();
			instance = dataTest.instance(6);

			distances = kmedoids.distanceToMedoids(instance);
			System.out.println("\tDistance to medoids: \n" + weka.core.Utils.arrayToString(distances));

			distribution = kmedoids.distributionForInstance(instance);
			System.out.println("\tDistribution: \n" + weka.core.Utils.arrayToString(distribution));

			System.out.println("\tInstance assigned to Cluster (0, " + (kmedoids.numberOfClusters() - 1) + "): "
					+ kmedoids.clusterInstance(instance));

		} catch (Exception e) {

			e.printStackTrace();
		}
		System.out.println("\nThe program finished normally.");

	}

}
