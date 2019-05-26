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
package miml.transformation.mimlTOmi;

import java.io.Serializable;

import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import mulan.transformations.BinaryRelevanceTransformation;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 
 * Class that uses Binary Relevance transformation to convert MIMLInstances to
 * MIL Instances with relational attribute.
 * 
 * 
 * @author Ana I. Reyes Melero
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20170507
 *
 */
public class BRTransformation implements Serializable {

	/** For serialization */
	private static final long serialVersionUID = -3662731281779529497L;

	/** Binary Relevance Transformation */
	protected BinaryRelevanceTransformation BRT;

	/** MIML dataSet */
	protected MIMLInstances dataSet;

	/**
	 * Constructor.
	 * 
	 *
	 * @param dataSet MIMLInstances dataset
	 */
	public BRTransformation(MIMLInstances dataSet) {
		this.dataSet = dataSet;
		this.BRT = new BinaryRelevanceTransformation(dataSet);
	}

	/**
	 * Removes all label attributes except labelToKeep.
	 *
	 * @param instance    The instance from which labels are to be removed.
	 * @param labelToKeep The label to keep. A value in [0, numLabels-1].
	 * @return Instance
	 */
	public Instance transformBag(MIMLBag instance, int labelToKeep) {
		return BRT.transformInstance(instance, labelToKeep);
	}

	/**
	 * Removes all label attributes except labelToKeep.
	 *
	 * @param bagIndex    The bagIndex of the Bag to be transformed.
	 * @param labelToKeep The label to keep. A value in [0, numLabels-1].
	 * @return Instance
	 * @throws Exception To be handled in upper level.
	 */
	public Instance transformBag(int bagIndex, int labelToKeep) throws Exception {
		return BRT.transformInstance(dataSet.getBag(bagIndex), labelToKeep);
	}

	/**
	 * Remove all label attributes except label at position indexToKeep.
	 *
	 * @param instance     The instance from which labels are to be removed.
	 * @param labelIndices Array storing, for each label its corresponding label
	 *                     index.
	 * @param indexToKeep  The label index to keep.
	 * @return transformed Instance.
	 * 
	 */
	public static Instance transformBag(MIMLBag instance, int[] labelIndices, int indexToKeep) {
		return BinaryRelevanceTransformation.transformInstance(instance, labelIndices, indexToKeep);
	}

	/**
	 * Remove all label attributes except labelToKeep.
	 *
	 * @param labelToKeep The label to keep. A value in [0, numLabels-1].
	 * @return Instances
	 * @throws Exception To be handled in an upper level.
	 */
	public Instances transformBags(int labelToKeep) throws Exception {
		return BRT.transformInstances(labelToKeep);
	}

	/**
	 * Remove all label attributes except that at indexOfLabelToKeep
	 *
	 * @param dataSet      A MIMLInstances dataset.
	 * @param labelIndices Array storing, for each label its corresponding label
	 *                     index.
	 * @param indexToKeep  The label index to keep.
	 * @return Instances
	 * @throws Exception when removal fails.
	 */
	public static Instances transformBags(MIMLInstances dataSet, int[] labelIndices, int indexToKeep) throws Exception {
		return BinaryRelevanceTransformation.transformInstances(dataSet.getDataSet(), labelIndices, indexToKeep);
	}

}
