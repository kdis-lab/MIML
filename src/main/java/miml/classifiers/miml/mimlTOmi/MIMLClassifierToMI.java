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

import org.apache.commons.configuration2.Configuration;

import miml.classifiers.miml.MIMLClassifier;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.TransformationBasedMultiLabelLearner;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;

/**
 * 
 * <p>
 * Class implementing the degenerative algorithm for MIML data to solve it with
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
 *
 */
public class MIMLClassifierToMI extends MIMLClassifier {

	/** For serialization. */
	private static final long serialVersionUID = -1665460849023571048L;

	/** Generic classifier. */
	private MultiLabelLearner transformationClassifier;

	/**
	 * Constructor.
	 * 
	 * @param transformationClassifier Classifier
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
	protected MultiLabelOutput makePredictionInternal(MIMLBag instance) throws Exception, InvalidDataException {
		return transformationClassifier.makePrediction(instance);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * core.IConfiguration#configure(org.apache.commons.configuration.Configuration)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void configure(Configuration configuration) {

		try {

			// Get the transformation classifier method
			String transformName = configuration.getString("transformMethod[@name]");
			// Instantiate the transformation classifier class used in the experiment
			Class<? extends MultiLabelLearner> clsClass = (Class<? extends MultiLabelLearner>) Class
					.forName(transformName);

			// Get the name of the MI base classifier class
			String baseName = configuration.getString("multiInstanceClassifier[@name]");
			// Instance class
			Class<? extends Classifier> baseClassifier = (Class<? extends Classifier>) Class.forName(baseName);
			// Check if options is setted
			String optionsAux = configuration.subset("multiInstanceClassifier").getString("listOptions");

			if (optionsAux != null) {
				String[] options = optionsAux.split(" ");

				Classifier classifier = baseClassifier.newInstance();

				((AbstractClassifier) classifier).setOptions(options);

				transformationClassifier = clsClass.getConstructor(Classifier.class).newInstance(classifier);
			} else {
				transformationClassifier = clsClass.getConstructor(Classifier.class)
						.newInstance(baseClassifier.newInstance());
			}

			if (!(transformationClassifier instanceof TransformationBasedMultiLabelLearner)) {
				throw new Exception(
						"Transformation method must be a instance of TransformationBasedMultiLabelLearner class");
			}

		} catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}

	}
}
