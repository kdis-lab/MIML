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

import miml.data.MIMLInstances;
import miml.data.MLSave;
import miml.transformation.mimlTOmi.BRTransformation;
import miml.transformation.mimlTOmi.LPTransformation;
import weka.core.Instances;
import weka.core.Utils;

/**
 * 
 * Class for basic handling of MIML to MIL LP and BR transformation.
 * 
 * @author Ana I. Reyes Melero
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20170507
 *
 */
public class MIMLtoMITranformation {
	/** Shows the help on command line. */
	public static void showUse() {
		System.out.println("Program parameters:");
		System.out.println("\t-f arffPathFile Name -> path of arff source file.");
		System.out.println("\t-o arffPathFile Name -> path of arff output file.");
		System.out.println("\t-x xmlPathFileName -> path of xml file.");
		System.out.println("Example:");
		System.out.println("\tjava -jar exampleMIMLtoMILTransformation -f data" + File.separator + "toy.arff -x data"
				+ File.separator + "toy.xml -o data" + File.separator + "toyResult.arff");
		System.exit(-1);
	}

	public static void main(String[] args) throws Exception {

		String arffFileName = Utils.getOption("f", args);
		String xmlFileName = Utils.getOption("x", args);
		String arffFileResult = Utils.getOption("o", args);

		// Parameter checking
		if (arffFileName.isEmpty()) {
			System.out.println("Arff pathName of source file must be specified.");
			showUse();
		}
		if (arffFileResult.isEmpty()) {
			System.out.println("Arff pathName of output file must be specified.");
			showUse();
		}
		if (xmlFileName.isEmpty()) {
			System.out.println("Xml pathName must be specified.");
			showUse();
		}

		// Loads the dataset
		System.out.println("Loading the dataset....");

		MIMLInstances mimlDataSet = new MIMLInstances(arffFileName, xmlFileName);

		System.out.println("===================Label Powerset=====================");
		LPTransformation lp = new LPTransformation();
		Instances transformed = lp.transformBags(mimlDataSet);
		MLSave.saveArff(transformed, arffFileResult);

		System.out.println("===================Binary Relevance=====================");
		BRTransformation br = new BRTransformation(mimlDataSet);
		for (int i = 0; i < mimlDataSet.getNumLabels(); i++) {
			transformed = br.transformBags(i);
			MLSave.saveArff(transformed, i + arffFileResult);
		}

		// Saves arff file
		MLSave.saveXml(mimlDataSet, xmlFileName);

		System.out.println("The program has finished.");
	}

}
