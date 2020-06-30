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

package miml.classifiers.miml.mimlTOmi;

import miml.classifiers.miml.MIMLClassifier;
import miml.core.ConfigParameters;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.TransformationBasedMultiLabelLearner;
import mulan.data.MultiLabelInstances;
import org.apache.commons.configuration2.Configuration;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Utils;

import java.util.Objects;

/**
 * <p>
 * Class implementing the transformation algorithm for MIML data to solve it with
 * MI learning. For more information, see <em>Zhou, Z. H., &#38; Zhang, M. L.
 * (2007). Multi-instance multi-label learning with application to scene
 * classification. In Advances in neural information processing systems (pp.
 * 1609-1616).</em>
 * </p>
 *
 * @author Alvaro A. Belmonte
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20180701
 */
public class MIMLClassifierToMI extends MIMLClassifier {

	/**
	 * Generated Serial version UID.
	 */
	private static final long serialVersionUID = -1665460849023571048L;

	/**
	 * Generic classifier used for transformation.
	 */
	protected MultiLabelLearner transformationClassifier;

	/**
	 * Basic constructor.
	 *
	 * @param transformationClassifier Mulan MultiLabelLearner used as
	 *                                 transformation method from MIML to MI.
	 */
	public MIMLClassifierToMI(MultiLabelLearner transformationClassifier) {
		super();
		this.transformationClassifier = transformationClassifier;
	}

	/**
	 * No-argument constructor for xml configuration.
	 */
	public MIMLClassifierToMI() {
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see mimlclassifier.MIMLClassifier#buildInternal(data.MIMLInstances)
	 */
	@Override
	protected void buildInternal(MIMLInstances trainingSet) throws Exception {
		MultiLabelInstances mlData = new MultiLabelInstances(trainingSet.getDataSet(), trainingSet.getLabelsMetaData());
		transformationClassifier.setDebug(getDebug());
		transformationClassifier.build(mlData);
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see mimlclassifier.MIMLClassifier#makePredictionInternal(data.Bag)
	 */
	@Override
	protected MultiLabelOutput makePredictionInternal(MIMLBag instance) throws Exception {
		return transformationClassifier.makePrediction(instance);
	}

	/*
	 * (non-Javadoc)
	 *
	 * @see
	 * core.IConfiguration#configure(org.apache.commons.configuration.Configuration)
	 */
	@Override
	public void configure(Configuration configuration) {
		// Get the transformation classifier method
		String transformName = configuration.getString("transformationMethod[@name]");
		// Instantiate the transformation classifier class used in the experiment
		Class<? extends MultiLabelLearner> clsClass = null;
		try {
			clsClass = Class.forName(transformName).asSubclass(MultiLabelLearner.class);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}

		// Get the name of the MI base classifier class
		String baseName = configuration.getString("multiInstanceClassifier[@name]");
		// Instance class
		Class<? extends Classifier> baseClassifier = null;
		try {
			baseClassifier = Class.forName(baseName).asSubclass(Classifier.class);
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		}

		// Check if options is setted
		String strOptions = configuration.subset("multiInstanceClassifier").getString("listOptions");
		if (strOptions != null) {
			try {
				Classifier classifier = Objects.requireNonNull(baseClassifier).getConstructor().newInstance();
				((AbstractClassifier) classifier).setOptions(Utils.splitOptions(strOptions));
				transformationClassifier = Objects.requireNonNull(clsClass).getConstructor(Classifier.class)
						.newInstance(classifier);
			} catch (Exception e) {
				e.printStackTrace();
				System.exit(1);
			}
		} else {
			try {
				transformationClassifier = Objects.requireNonNull(clsClass).getConstructor(Classifier.class)
						.newInstance(Objects.requireNonNull(baseClassifier).getConstructor().newInstance());
			} catch (Exception e) {
				e.printStackTrace();
				System.exit(1);
			}
		}

		if (!(transformationClassifier instanceof TransformationBasedMultiLabelLearner)) {
			try {
				throw new Exception(
						"Transformation method must be an instance of TransformationBasedMultiLabelLearner class");
			} catch (Exception e) {
				e.printStackTrace();
				System.exit(1);
			}
		}

		ConfigParameters.setClassifierName(baseName);
		ConfigParameters.setTransformationMethod(transformName);
		ConfigParameters.setIsTransformation(true);
	}
}
