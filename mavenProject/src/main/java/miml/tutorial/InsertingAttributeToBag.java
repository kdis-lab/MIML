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
import java.util.ArrayList;

import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import miml.data.MLSave;
import weka.core.Attribute;
import weka.core.Utils;

/**
 * 
 * Class implementing an example of inserting a new attribute to the relational attribute of the
 * dataset with {0,1} values.
 * 
 * @author Alvaro A. Belmonte
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20190525
 *
 */
public class InsertingAttributeToBag {
	
	/** Shows the help on command line. */
	public static void showUse() {
		System.out.println("Program parameters:");
		System.out.println("\t-f arffPathFile Name -> path of arff source file.");
		System.out.println("\t-x xmlPathFileName -> path of xml file.");
		System.out.println("Example:");
		System.out.println("\tjava -jar InsertingAttributesToBag -f data" + File.separator + "toy.arff -x data"
				+ File.separator + "toy.xml");
		System.exit(-1);
	}
	
	public static void main(String[] args) throws Exception {

		//-f data/miml_birds.arff -x data/miml_birds.xml
		
		String arffFileName = Utils.getOption("f", args);
		String xmlFileName = Utils.getOption("x", args);

		// Loads the dataset
		System.out.println("Loading the dataset....");
		MIMLInstances mimlDataSet = new MIMLInstances(arffFileName, xmlFileName);

		String newName = new String("NewAttrName");
		ArrayList<String> values = new ArrayList<String>();
		values.add("0");
		values.add("1");
		Attribute attr = new Attribute(newName, values);

		//This inserts the attribute into bags and sets it as '?' for all instances and all bags
		MIMLInstances result = mimlDataSet.insertAttributeToBags(attr);		
		
		// provides {0,1} values for the attribute added
		int value = 0;
		for (int i = 0; i < result.getNumBags(); i++) {
			MIMLBag bag = result.getBag(i);
			for (int j = 0; j < bag.getNumInstances(); j++) {
				bag.setValue(j, mimlDataSet.getNumAttributesInABag(), value % 2);
				value++;
			}
		}
		MLSave.saveArff(result.getDataSet(), "data" + File.separator + "new.arff");
		
	}
	
}
