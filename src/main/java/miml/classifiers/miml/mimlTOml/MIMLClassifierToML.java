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

import org.apache.commons.configuration2.Configuration;

import miml.classifiers.miml.MIMLClassifier;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import miml.transformation.mimlTOml.MIMLtoML;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;

/**
 * 
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
 *
 */
public class MIMLClassifierToML extends MIMLClassifier {

	/** For serialization. */
	private static final long serialVersionUID = 1L;

	/** A Generic MultiLabel classifier. */
	MultiLabelLearner baseClassifier;

	/** The transform method. */
	MIMLtoML transformMethod;

	/** The miml dataset. */
	MIMLInstances mimlDataset;

	/**
	 * Constructor.
	 *
	 * @param baseClassifier  Classifier
	 * @param transformMethod the transform method
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
	protected MultiLabelOutput makePredictionInternal(MIMLBag bag) throws Exception, InvalidDataException {
		Instance instance = transformMethod.transformInstance(bag);
		return baseClassifier.makePrediction(instance);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * core.IConfiguration#configure(org.apache.commons.configuration.Configuration)
	 */
	@SuppressWarnings({ "unchecked", "rawtypes" })
	@Override
	public void configure(Configuration configuration) {

		try {
			// Get the string with the base classifier class
			String classifierName = configuration.getString("multiLabelClassifier[@name]");
			// Instance class
			Class<? extends MultiLabelLearner> classifierClass = (Class<? extends MultiLabelLearner>) Class
					.forName(classifierName);
			Configuration subConfiguration = configuration.subset("multiLabelClassifier"); // getProperty("multiLable")
			// Parameters length
			int parameterLength = subConfiguration.getList("parameters.classParameters").size();

			// Obtaining las clasess
			Class[] cArg = new Class[parameterLength];
			Object[] obj = new Object[parameterLength];

			String parameter;

			for (int i = 0; i < parameterLength; i++) {

				parameter = configuration.getString("multiLabelClassifier.parameters.classParameters(" + i + ")");

				if (parameter.equals("int.class")) {
					cArg[i] = int.class;
					obj[i] = configuration.getInt("multiLabelClassifier.parameters.valueParameters(" + i + ")");

				} else if (parameter.equals("double.class")) {
					cArg[i] = double.class;
					obj[i] = configuration.getDouble("multiLabelClassifier.parameters.valueParameters(" + i + ")");

				} else if (parameter.equals("char.class")) {
					cArg[i] = char.class;
					obj[i] = configuration.getInt("multiLabelClassifier.parameters.valueParameters(" + i + ")");

				} else if (parameter.equals("byte.class")) {
					cArg[i] = byte.class;
					obj[i] = configuration.getByte("multiLabelClassifier.parameters.valueParameters(" + i + ")");

				} else if (parameter.equals("boolean.class")) {
					cArg[i] = boolean.class;
					obj[i] = configuration.getBoolean("multiLabelClassifier.parameters.valueParameters(" + i + ")");

				} else if (parameter.equals("String.class")) {
					cArg[i] = String.class;
					obj[i] = configuration.getString("multiLabelClassifier.parameters.valueParameters(" + i + ")");

				}
				// Here you can add the rest of types (short, long, ...)
				else {
					cArg[i] = Class.forName(parameter);
					obj[i] = Class
							.forName(configuration
									.getString("multiLabelClassifier.parameters.valueParameters(" + i + ")"))
							.newInstance();
				}

			}
			
			this.baseClassifier = (MultiLabelLearner) classifierClass.getConstructor(cArg).newInstance(obj);

			// Get the string with the base classifier class
			String transformerName = configuration.getString("transformMethod[@name]");
			// Instance class
			Class<? extends MIMLtoML> transformerClass = (Class<? extends MIMLtoML>) Class.forName(transformerName);
			this.transformMethod = transformerClass.newInstance();

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}

	}

}
