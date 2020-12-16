package miml.classifiers.miml.mimlTOmi;

import mulan.classifier.transformation.BinaryRelevance;
import weka.classifiers.Classifier;

/**
 * Wrapper for mulan BinaryRelevance to be used in MIML to MI algorithms.
 * 
 * @author Alvaro A. Belmonte
 * @author Amelia Zafra
 * @author Eva Gigaja
 * @version 20180619
 */
public class MIMLBinaryRelevance extends BinaryRelevance{

	/**
	 * Generated Serial version UID.
	 */
	private static final long serialVersionUID = 1706817441965109002L;

    /**
     * Creates a new instance.
     *
     * @param classifier The base-level classification algorithm that will be
     * used for training each of the binary models.
     */
	public MIMLBinaryRelevance(Classifier classifier) {
		super(classifier);
		// TODO Auto-generated constructor stub
	}

}
