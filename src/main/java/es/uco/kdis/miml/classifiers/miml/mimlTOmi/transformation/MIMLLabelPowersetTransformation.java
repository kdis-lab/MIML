package es.uco.kdis.miml.classifiers.miml.mimlTOmi.transformation;

import mulan.transformations.LabelPowersetTransformation;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class MIMLLabelPowersetTransformation extends LabelPowersetTransformation {

	/**
	 * 
	 */
	private static final long serialVersionUID = -515679901670889755L;

	/**
	 * 
	 * @param instance     the instance to be transformed
	 * @param labelIndices the labels to remove.
	 * @return tranformed instance
	 * @throws Exception Potential exception thrown. To be handled in an upper
	 *                   level.
	 */
	public Instance transformInstance(Instance instance, int[] labelIndices) throws Exception {

		// Prepares a dataset with a single instance
		Instances aux = new Instances(instance.dataset(), 0);
		aux.add(instance);

		// Remove labels
		Remove remove = new Remove();
		remove.setAttributeIndicesArray(labelIndices);
		remove.setInputFormat(aux);
		Instances result = Filter.useFilter(aux, remove);

		// Adds class attribute
		result.insertAttributeAt(getTransformedFormat().attribute(getTransformedFormat().classIndex()),
				result.numAttributes());
		result.setClassIndex(result.numAttributes() - 1);

		return result.instance(0);

	}

}
