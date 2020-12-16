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

import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import miml.data.statistics.MIMLStatistics;
import mulan.data.InvalidDataFormatException;
import weka.core.Instance;
import weka.core.Utils;

/**
 * 
 * Class implementing basic handling of MIML datasets.
 * 
 * @author Ana I.Reyes Melero
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @author Alvaro A. Belmonte
 * @version 20190525
 *
 */

public class ManagingMIMLInstances {

	/** Shows the help on command line. */
	public static void showUse() {
		System.out.println("Program parameters:");
		System.out.println("\t-f arffPathFile Name -> path of arff file.");
		System.out.println("\t-x xmlPathFileName -> path of arff file.");
		System.out.println("Example:");
		System.out.println("\tjava -jar example MIMLInstances -f data" + File.separator + "toy.arff -x data"
				+ File.separator + "toy.xml");
		System.exit(-1);
	}

	public static void main(String[] args) {

		try {

			String arffFileName = Utils.getOption("f", args);
			String xmlFileName = Utils.getOption("x", args);

			// Parameter checking
			if (arffFileName.isEmpty()) {
				System.out.println("Arff pathName must be specified.");
				showUse();
			}
			if (xmlFileName.isEmpty()) {
				System.out.println("Xml pathName must be specified.");
				showUse();
			}

			// Loads the dataset
			System.out.println("Loading the dataset....");
			MIMLInstances mimlDataSet = new MIMLInstances(arffFileName, xmlFileName);

			System.out.println("Number of bags: " + mimlDataSet.getNumBags());
			System.out.println("Number of Instances (bags): " + mimlDataSet.getNumInstances());
			System.out.println("Number of Labels: " + mimlDataSet.getNumLabels());
			System.out.println("Number of Attributes: " + mimlDataSet.getNumAttributes());
			System.out.println("Number of AttributesPerBag: " + mimlDataSet.getNumAttributesInABag());
			System.out.println("Number of AttributesWithRelational: " + mimlDataSet.getNumAttributesWithRelational());

			// Shows all bags in the dataset
			for (int i = 0; i < mimlDataSet.getNumBags(); i++) {
				MIMLBag bag = null;

				// Recover a bag
				bag = mimlDataSet.getBag(i);
				System.out.println("Bag: " + i);
				System.out.println("\tNumInstances: " + bag.getNumInstances());
				System.out.println("\tNumAttributes: " + bag.numAttributes());
				System.out.println("\tAttributesInABag: " + bag.getNumAttributesInABag());
				System.out.println("\tAttributesWithRelational: " + bag.getNumAttributesWithRelational());

				// Shows all instances in the bag
				for (int j = 0; j < bag.getNumInstances(); j++) {
					// Recovers an instance
					Instance instance = mimlDataSet.getInstance(i, j);
					System.out.println("\t\tInstance: " + j + " NumAttributes: " + instance.numAttributes());
					for (int k = 0; k < instance.numAttributes(); k++)
						System.out.println("\t\t\tAttribute " + k + ": " + instance.value(k));
				}
			}

			// Shows MIML metrics
			MIMLStatistics statsMIML = new MIMLStatistics(mimlDataSet);
			System.out.println(statsMIML.toString());

		} catch (InvalidDataFormatException e) {
			e.printStackTrace();
		} catch (Exception e) {

			e.printStackTrace();
		}

	}

}
