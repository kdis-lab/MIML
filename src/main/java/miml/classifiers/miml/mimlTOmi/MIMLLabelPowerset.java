package miml.classifiers.miml.mimlTOmi;

import miml.transformation.mimlTOmi.LPTransformation;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.core.Instances;

public class MIMLLabelPowerset extends LabelPowerset {

	/**
	 * 
	 */
	private static final long serialVersionUID = -515679901670889755L;

	public MIMLLabelPowerset(Classifier classifier) {
		super(classifier);
	}

	protected void buildInternal(MultiLabelInstances mlData) throws Exception {
		Instances transformedData;
		LPTransformation lp = new LPTransformation();
		transformation = lp.getLPT();
		debug("Transforming the training set.");
		transformedData = transformation.transformInstances(mlData);

		// debug("Transformed training set: \n + transformedData.toString());

		// check for unary class
		debug("Building single-label classifier.");
		if (transformedData.attribute(transformedData.numAttributes() - 1).numValues() > 1) {
			baseClassifier.buildClassifier(transformedData);
		}
	}

}
