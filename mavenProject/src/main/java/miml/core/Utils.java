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

package miml.core;

import java.util.ArrayList;
import org.apache.commons.configuration2.Configuration;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Resample;

/**
 * 
 * This class has utilies that can be used anywhere in the library.
 * 
 * @author Aurora Esteban Toscano
 * @author Alvaro A. Belmonte
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20220613
 *
 */
public final class Utils {

	/**
	 * Obtains a sample of the original data.
	 *
	 * @param data                  Instances with the dataset.
	 * @param percentage            percentage of instances that will contain the
	 *                              new dataset.
	 * @param sampleWithReplacement If true the sampling will be with replacement.
	 * @param seed                  Seed for randomization. Necessary if instances
	 *                              have not been previously shuffled with
	 *                              randomize.
	 * 
	 * @return Instances.
	 * @throws Exception To be handled.
	 */
	public static Instances resample(Instances data, double percentage, boolean sampleWithReplacement, int seed)
			throws Exception {

		Instances resampled;

		Resample rspl = new Resample();
		rspl.setRandomSeed(seed);
		rspl.setSampleSizePercent(percentage);
		rspl.setNoReplacement(!sampleWithReplacement);
		rspl.setInputFormat(data);
		resampled = Filter.useFilter(data, rspl);

		return resampled;
	}

	/**
	 * Read the configuration parameters for a specific Multi Label classifier's
	 * constructor
	 * 
	 * @param configuration Configuration used to configure the class
	 * @return Params class which contains the parameters of classifier's
	 *         constructor
	 */
	@SuppressWarnings("unchecked")
	public static Params readMultiLabelLearnerParams(Configuration configuration) {
		int nParams = configuration.getList("parameters.parameter[@class]").size();
		Class<?>[] classes = new Class[nParams];
		Object[] objects = new Object[nParams];

		for (int i = 0; i < nParams; i++) {
			Params subparams = null;
			if (configuration.getList("parameters.parameter(" + i + ").parameters.parameter[@class]", new ArrayList<>())
					.size() > 0)
				subparams = readMultiLabelLearnerParams(configuration.subset("parameters.parameter(" + i + ")"));

			String className = configuration.getString("parameters.parameter(" + i + ")[@class]");

			switch (className) {
			case "int.class":
				classes[i] = int.class;
				objects[i] = configuration.getInt("parameters.parameter(" + i + ")[@value]");
				break;
			case "double.class":
				classes[i] = double.class;
				objects[i] = configuration.getDouble("parameters.parameter(" + i + ")[@value]");
				break;
			case "char.class":
				classes[i] = char.class;
				objects[i] = configuration.getInt("parameters.parameter(" + i + ")[@value]");
				break;
			case "byte.class":
				classes[i] = byte.class;
				objects[i] = configuration.getByte("parameters.parameter(" + i + ")[@value]");
				break;
			case "boolean.class":
				classes[i] = boolean.class;
				objects[i] = configuration.getBoolean("parameters.parameter(" + i + ")[@value]");
				break;
			case "String.class":
				classes[i] = String.class;
				objects[i] = configuration.getString("parameters.parameter(" + i + ")[@value]");
				break;
			case "short.class":
				classes[i] = short.class;
				objects[i] = configuration.getShort("parameters.parameter(" + i + ")[@value]");
				break;
			case "long.class":
				classes[i] = long.class;
				objects[i] = configuration.getLong("parameters.parameter(" + i + ")[@value]");
				break;
			default:
				try { 
					classes[i] = Class.forName(className);
					if (classes[i].isEnum()) {
						objects[i] = Enum.valueOf(classes[i].asSubclass(Enum.class),
								configuration.getString("parameters.parameter(" + i + ")[@value]"));
					} else { 
						//The parameter is a weka algorithm						
						if (subparams == null) {
							objects[i] = Class
									.forName(configuration.getString("parameters.parameter(" + i + ")[@value]"))
									.getConstructor().newInstance();							
						} else {
							objects[i] = Class
									.forName(configuration.getString("parameters.parameter(" + i + ")[@value]"))
									.getConstructor(subparams.getClasses()).newInstance(subparams.getObjects());
						}
						String options = configuration.getString("parameters.parameter(" + i + ")[@listOptions]");
						if(options!=null) //There are options for the weka classifier
							   ((AbstractClassifier)objects[i]).setOptions(weka.core.Utils.splitOptions(options));

					}
				} catch (Exception e) {
					e.printStackTrace();
					System.exit(1);
				}
				break;
			}
		}
		return new Params(classes, objects);
	}

}
