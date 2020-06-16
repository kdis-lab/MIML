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

package miml.classifiers.miml.mimlTOml;

import miml.classifiers.miml.MIMLClassifier;
import miml.core.ConfigParameters;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import miml.transformation.mimlTOml.MIMLtoML;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import org.apache.commons.configuration2.Configuration;
import weka.core.Instance;

import java.util.ArrayList;
import java.util.Objects;

/**
 * <p>
 * Class implementing the degenerative algorithm for MIML data to solve it with
 * ML learning. For more information, see <em>Zhou, Z. H., &#38; Zhang, M. L.
 * (2007). Multi-instance multi-label learning with application to scene
 * classification. In Advances in neural information processing systems (pp.
 * 1609-1616).</em>
 * </p>
 *
 * @author Alvaro A. Belmonte
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20180608
 */
public class MIMLClassifierToML extends MIMLClassifier {

	/**
	 * Generated Serial version UID.
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * A Generic MultiLabel classifier.
	 */
	protected MultiLabelLearner baseClassifier;

	/**
	 * The transform method.
	 */
	protected MIMLtoML transformMethod;

	/**
	 * The miml dataset.
	 */
	protected MIMLInstances mimlDataset;

	/**
	 * Basic constructor to initialize the classifier.
	 *
	 * @param baseClassifier  The base classification algorithm.
	 * @param transformMethod Algorithm used as transformation method from MIML to ML.
	 * @throws Exception To be handled in an upper level.
	 */
	public MIMLClassifierToML(MultiLabelLearner baseClassifier, MIMLtoML transformMethod) throws Exception {
		super();
		this.baseClassifier = baseClassifier;
		this.transformMethod = transformMethod;
	}

	/**
	 * No-argument constructor for xml configuration.
	 */
	public MIMLClassifierToML() {
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see mimlclassifier.MIMLClassifier#buildInternal(data.MIMLInstances)
	 */
	@Override
	public void buildInternal(MIMLInstances mimlDataSet) throws Exception {
		// Transforms a dataset
		MultiLabelInstances mlDataSet = transformMethod.transformDataset(mimlDataSet);
		baseClassifier.build(mlDataSet);
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see mimlclassifier.MIMLClassifier#makePredictionInternal(data.Bag)
	 */
	@Override
	protected MultiLabelOutput makePredictionInternal(MIMLBag bag) throws Exception {
		Instance instance = transformMethod.transformInstance(bag);
		return baseClassifier.makePrediction(instance);
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see
	 * core.IConfiguration#configure(org.apache.commons.configuration.Configuration)
	 */
	@Override
	public void configure(Configuration configuration) {
		// Get the string with the base classifier class
		String classifierName = configuration.getString("multiLabelClassifier[@name]");
		// Instance class
		Class<? extends MultiLabelLearner> classifierClass = null;
		try {
			classifierClass = Class.forName(classifierName).asSubclass(MultiLabelLearner.class);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		}

		Params params = readParams(configuration.subset("multiLabelClassifier"));

		try {
			this.baseClassifier = Objects.requireNonNull(classifierClass).getConstructor(params.classes).newInstance(params.objects);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}

		// Get the string with the base classifier class
		String transformerName = configuration.getString("transformMethod[@name]");
		// Instance class
		Class<? extends MIMLtoML> transformerClass = null;
		try {
			transformerClass = Class.forName(transformerName).asSubclass(MIMLtoML.class);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		}
		try {
			this.transformMethod = Objects.requireNonNull(transformerClass).getConstructor().newInstance();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}

		ConfigParameters.setClassifierName(classifierName);
		ConfigParameters.setTransformMethod(transformerName);
		ConfigParameters.setIsDegenerative(true);
	}

	private Params readParams(Configuration configuration) {
		int nParams = configuration.getList("parameters.parameter[@class]").size();
		Class<?>[] classes = new Class[nParams];
		Object[] objects = new Object[nParams];

		for (int i = 0; i < nParams; i++) {
			Params subparams = null;
			if (configuration.getList("parameters.parameter(" + i + ").parameters.parameter[@class]",
					new ArrayList<>()).size() > 0)
				subparams = readParams(configuration.subset("parameters.parameter(" + i + ")"));

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
							objects[i] = Enum.valueOf(classes[i].asSubclass(Enum.class), configuration.
									getString("parameters.parameter(" + i + ")[@value]"));
						} else {
							if (subparams == null) {
								objects[i] = Class.forName(configuration
										.getString("parameters.parameter(" + i + ")[@value]"))
										.getConstructor().newInstance();
							} else {
								objects[i] = Class.forName(configuration
										.getString("parameters.parameter(" + i + ")[@value]"))
										.getConstructor(subparams.classes).newInstance(subparams.objects);
							}
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

	private static class Params {
		private final Class<?>[] classes;
		private final Object[] objects;

		public Params(Class<?>[] classes, Object[] objects) {
			this.classes = classes;
			this.objects = objects;
		}
	}
}
