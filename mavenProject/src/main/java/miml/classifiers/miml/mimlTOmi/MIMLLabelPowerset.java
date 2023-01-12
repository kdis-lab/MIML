package miml.classifiers.miml.mimlTOmi;

import miml.transformation.mimlTOmi.LPTransformation;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Wrapper for mulan LabelPowerset to be used in MIML to MI algorithms.
 * 
 * @author Alvaro A. Belmonte
 * @author Amelia Zafra
 * @author Eva Gigaja
 * @version 20180619
 */
public class MIMLLabelPowerset extends LabelPowerset {

	/**
	 * Generated Serial version UID.
	 */
	private static final long serialVersionUID = -515679901670889755L;

	/**
	 * Constructor that initializes the learner with a base classifier.
	 *
	 * @param classifier The base single-label classification algorithm.
	 */
	public MIMLLabelPowerset(Classifier classifier) {
		super(classifier);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see mulan.classifier.transformation.LabelPowerset#buildInternal(mulan.data.
	 * MultiLabelInstances)
	 */
	@Override
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
