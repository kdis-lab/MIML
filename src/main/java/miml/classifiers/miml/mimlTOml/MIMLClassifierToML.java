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

import java.util.Objects;

import org.apache.commons.configuration2.Configuration;

import miml.classifiers.miml.MIMLClassifier;
import miml.core.ConfigParameters;
import miml.core.Params;
import miml.core.Utils;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import miml.transformation.mimlTOml.MIMLtoML;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;

/**
 * <p>
 * Class implementing the transformation algorithm for MIML data to solve it with
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
	protected MIMLtoML transformationMethod;

	/**
	 * The miml dataset.
	 */
	protected MIMLInstances mimlDataset;

	/**
	 * Basic constructor to initialize the classifier.
	 *
	 * @param baseClassifier  The base classification algorithm.
	 * @param transformationMethod Algorithm used as transformation method from MIML to
	 *                        ML.
	 * @throws Exception To be handled in an upper level.
	 */
	public MIMLClassifierToML(MultiLabelLearner baseClassifier, MIMLtoML transformationMethod) throws Exception {
		super();
		this.baseClassifier = baseClassifier;
		this.transformationMethod = transformationMethod;
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
		MultiLabelInstances mlDataSet = transformationMethod.transformDataset(mimlDataSet);
		baseClassifier.build(mlDataSet);
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see mimlclassifier.MIMLClassifier#makePredictionInternal(data.Bag)
	 */
	@Override
	protected MultiLabelOutput makePredictionInternal(MIMLBag bag) throws Exception {
		Instance instance = transformationMethod.transformInstance(bag);
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

		Params params = Utils.readMultiLabelLearnerParams(configuration.subset("multiLabelClassifier"));

		try {
			this.baseClassifier = Objects.requireNonNull(classifierClass).getConstructor(params.getClasses())
					.newInstance(params.getObjects());
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}

		// Get the string with the base classifier class
		String transformerName = configuration.getString("transformationMethod[@name]");
		// Instance class
		Class<? extends MIMLtoML> transformerClass = null;
		try {
			transformerClass = Class.forName(transformerName).asSubclass(MIMLtoML.class);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
			System.exit(1);
		}
		try {
			this.transformationMethod = Objects.requireNonNull(transformerClass).getConstructor().newInstance();
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}

		ConfigParameters.setClassifierName(classifierName);
		ConfigParameters.setTransformationMethod(transformerName);
		ConfigParameters.setIsTransformation(true);
	}

}
