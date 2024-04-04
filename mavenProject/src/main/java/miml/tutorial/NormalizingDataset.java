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
import miml.data.normalization.MinMaxNormalization;

/**
 * 
 * Class to show an example of normalization of a MIML Dataset.
 * 
 * @author Eva Gibaja
 * @version 20220727
 *
 */
public class NormalizingDataset {

	public static void main(String[] args) {
		MinMaxNormalization norm = new MinMaxNormalization();
		MIMLInstances mimlDataSet1;
		MIMLInstances mimlDataSet2;
		try {
			System.out.println("Loading datasets");
			mimlDataSet1 = new MIMLInstances("data" + File.separator + "miml_birds_random_80train.arff",
					"data" + File.separator + "miml_birds.xml");
			mimlDataSet2 = new MIMLInstances("data" + File.separator + "miml_birds_random_20test.arff",
					"data" + File.separator + "miml_birds.xml");
			/*
			 * updateStats must be called before call normalize method. If several datasets
			 * with the same structure are normalized at once (e.g. train and test or folds
			 * partitioned files), this method will be called for each dataset before
			 * normalization. Besides, if the method method detects that all the attributes
			 * are yet normalized, it sets the "normalized" property as true.
			 */
			norm.updateStats(mimlDataSet1);
			norm.updateStats(mimlDataSet2);

			if (norm.isNormalized() == false) {
				norm.normalize(mimlDataSet1);
				norm.normalize(mimlDataSet2);
			}
		} catch (Exception e) {

			e.printStackTrace();
		}
		System.out.println("The datasets have been normalized");
	}// main

}
