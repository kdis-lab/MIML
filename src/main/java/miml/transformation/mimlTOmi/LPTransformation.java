package miml.transformation.mimlTOmi;

import java.io.Serializable;

import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import mulan.transformations.LabelPowersetTransformation;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * 
 * Class that uses LabelPowerset transformation to convert MIMLInstances to MIL
 * Instances with relational attribute.
 * 
 * @author Ana I. Reyes Melero
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20170507
 *
 */
public class LPTransformation implements Serializable {

	/** For serialization */
	private static final long serialVersionUID = -3418733531408587603L;

	/** LabelPowerSetTransformation */
	protected MIMLLabelPowersetTransformation LPT;

	/**
	 * Constructor
	 */
	public LPTransformation() {
		this.LPT = new MIMLLabelPowersetTransformation();
	}

	/**
	 * Returns the format of the transformed instances
	 * 
	 * @return the format of the transformed instances
	 */
	public LabelPowersetTransformation getLPT() {
		return LPT;
	}

	/**
	 * 
	 * @param bag          The bag to be transformed.
	 * @param labelIndices The labels to remove.
	 * @return Instance
	 * @throws Exception To be handled in an upper level.
	 */
	public Instance transformBag(MIMLBag bag, int[] labelIndices) throws Exception {
		return LPT.transformInstance(bag, labelIndices);
	}

	/**
	 * 
	 * @param dataSet MIMLInstances dataSet.
	 * @return Instances
	 * @throws Exception To be handled in an upper level.
	 */
	public Instances transformBags(MIMLInstances dataSet) throws Exception {
		return LPT.transformInstances(dataSet);
	}

}

class MIMLLabelPowersetTransformation extends LabelPowersetTransformation {

	/**
	 * 
	 */
	private static final long serialVersionUID = -515679901670889755L;

	/**
	 * 
	 * @param instance     the instance to be transformed
	 * @param labelIndices the labels to remove.
	 * @return transformed instance
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
