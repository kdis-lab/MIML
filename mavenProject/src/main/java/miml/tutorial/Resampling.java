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

import miml.core.Utils;
import miml.data.MIMLInstances;
import weka.core.Instances;

/**
 * 
 * Class to show an example of sampling with replacement.
 * 
 * @author Eva Gibaja
 * @version 20220727
 *
 */

public class Resampling {

	public static void main(String[] args) throws Exception {


		MIMLInstances mimlDataSet1 = new MIMLInstances("data" + File.separator + "miml_birds_random_80train.arff",
				"data" + File.separator + "miml_birds.xml");

		System.out.println("Loading original dataset with "+mimlDataSet1.getNumBags() + " bags");
		
		int seed = 1;
		boolean sampleWithReplacement = true;
		double samplePercentage = 80;

		Instances sample = Utils.resample(mimlDataSet1.getDataSet(), samplePercentage, sampleWithReplacement, seed);

		MIMLInstances mimlDataSet2 = new MIMLInstances(sample, mimlDataSet1.getLabelsMetaData());

		System.out.println("Resampled dataset with " + mimlDataSet2.getNumBags() + "/" + mimlDataSet1.getNumBags() + " bags");

	}

}
