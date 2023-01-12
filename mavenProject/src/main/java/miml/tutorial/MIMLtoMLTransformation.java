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
import miml.transformation.mimlTOml.ArithmeticTransformation;
import miml.transformation.mimlTOml.GeometricTransformation;
import miml.transformation.mimlTOml.MinMaxTransformation;
import miml.transformation.mimlTOml.PropositionalTransformation;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Utils;

/**
 * 
 * Class for basic handling of the transformation MIML to ML transformations.
 * 
 * @author Ana I. Reyes Melero
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20210604
 *
 */
public class MIMLtoMLTransformation {
	/** Shows the help on command line. */
	public static void showUse() {
		System.out.println("Program parameters:");
		System.out.println("\t-f arffPathFile Name -> path of arff source file.");
		System.out.println("\t-x xmlPathFileName -> path of xml file.");
		System.out.println("Example:");
		System.out.println("\t-f data" + File.separator + "toy.arff -x data" + File.separator + "toy.xml");
		System.exit(-1);
	}

	public static void main(String[] args) throws Exception {

		// -f data/miml_birds.arff -x data/miml_birds.xml

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

		System.out.println("=============Arithmetic=====================");
		String arffFileResultAri = "data" + File.separator + "resultAri.arff";
		String xmlFileResultAri = "data" + File.separator + "resultAri.xml";

		ArithmeticTransformation ari = new ArithmeticTransformation(mimlDataSet);
		// Transforms a single instance
		Instance instance = ari.transformInstance(mimlDataSet.getBag(0));
		instance.numAttributes();

		// Transforms a complete dataset
		MultiLabelInstances result = ari.transformDataset();
		MLSave.saveArff(result, arffFileResultAri);
		MLSave.saveXml(result, xmlFileResultAri);

		System.out.println("=============Geometric=====================");
		String arffFileResultGeo = "data" + File.separator + "resultGeo.arff";
		String xmlFileResultGeo = "data" + File.separator + "resultGeo.xml";

		GeometricTransformation geo = new GeometricTransformation(mimlDataSet);

		// Transforms a single instance
		instance = geo.transformInstance(mimlDataSet.getBag(0));
		instance.numAttributes();

		// Transforms a complete dataset
		result = geo.transformDataset();
		MLSave.saveArff(result, arffFileResultGeo);
		MLSave.saveXml(result, xmlFileResultGeo);

		System.out.println("=============MinMax=====================");
		String arffFileResultMinMax = "data" + File.separator + "resultMinMax.arff";
		String xmlFileResultMinMax = "data" + File.separator + "resultMinMax.xml";

		MinMaxTransformation miniMax = new MinMaxTransformation(mimlDataSet);

		// Transforms a single instance, returns a set of instances
		instance = miniMax.transformInstance(mimlDataSet.getBag(0));
		instance.numAttributes();

		// Transforms a complete dataset
		result = miniMax.transformDataset();
		MLSave.saveArff(result, arffFileResultMinMax);
		MLSave.saveXml(result, xmlFileResultMinMax);

		System.out.println("=============Propositional=====================");
		String arffFileResultPropositional = "data" + File.separator + "resultPropositional.arff";
		String xmlFileResultPropositional = "data" + File.separator + "resultPropositional.xml";

		PropositionalTransformation propositional = new PropositionalTransformation(mimlDataSet);
		propositional.setIncludeBagId(true); //by default the bagID attribute is not included
		// Transforms a single instance
		result = propositional.transformInstance(mimlDataSet.getBag(0));

		// Transforms a complete dataset
		result = propositional.transformDataset();
		MLSave.saveArff(result, arffFileResultPropositional);
		MLSave.saveXml(result, xmlFileResultPropositional);

	}

}
