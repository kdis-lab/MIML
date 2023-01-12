/*
 *    This program is free software; you can redistribute it and/or modify
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
package miml.data;

import com.mathworks.toolbox.javabuilder.*;

import weka.core.Instance;

/**
 * 
 * Class to serve as interface between MIMLInstances and Matlab data types.
 * 
 * @author Eva Gibaja
 * @version 20211024
 */
public class MWTranslator {

	/** A MIML dataset. */
	MIMLInstances mimlDataSet;

	/** Number of bags of the dataset */
	int nBags;

	/** Number of labels of the dataset */
	int nLabels;

	/** Number of attributes per bag */
	int attributesPerBag;

	/** Array with the attribute indices corresponding to the labels */
	int labelIndices[];

	/**
	 * Constructor.
	 * 
	 * @param mimlDataSet A MIML dataset.
	 */
	public MWTranslator(MIMLInstances mimlDataSet) {
		this.mimlDataSet = mimlDataSet;
		this.nBags = mimlDataSet.getNumBags();
		this.attributesPerBag = mimlDataSet.getNumAttributesInABag();
		this.nLabels = mimlDataSet.getMLDataSet().getNumLabels();
		this.labelIndices = mimlDataSet.getLabelIndices();
	}

	/**
	 * Returns all the bags in the MIMLInstances dataset in the format of a nBagsx1
	 * MWCellArray in which the ith bag is stored in aCellArray{i,1}. Each bag is a
	 * nInstxnAttributes array of double values.
	 * 
	 * @return Returns all the bags in the MIMLInstances dataset in the format of a
	 *         nBagsx1 MWCellArray in which the ith bag is stored in
	 *         aCellArray{i,1}. Each bag is a nInstxnAttributes array of double
	 *         values.
	 * @throws Exception To be handled.
	 */
	public MWCellArray getBags() throws Exception {

		MWCellArray aCellArrayIn = new MWCellArray(nBags, 1);

		for (int i = 0; i < nBags; i++) {

			MIMLBag bag = mimlDataSet.getBag(i);
			int nInst = bag.getNumInstances();

			double[][] values = new double[nInst][attributesPerBag];
			for (int j = 0; j < nInst; j++) {
				Instance instance = bag.getInstance(j);
				for (int k = 0; k < attributesPerBag; k++)
					values[j][k] = instance.value(k);
			}

			int index[] = new int[2];
			index[0] = (i + 1);
			index[1] = 1;
			aCellArrayIn.set(index, values);
		}

		return aCellArrayIn;
	}

	/**
	 * Returns label associations of all bags in the MIMLInstances dataset in the
	 * format of a nLabelsxnBags MWNumericArray of double. If the ith bag belongs to
	 * the jth label, then aDoubleArray(j,i) equals +1, otherwise train_target(j,i)
	 * equals -1.
	 * 
	 * @return Returns label associations of all bags in the MIMLInstances dataset
	 *         in the format of a nLabelsxnBags MWNumericArray of double. If the ith
	 *         bag belongs to the jth label, then aDoubleArray(j,i) equals +1,
	 *         otherwise train_target(j,i) equals -1.
	 * @throws Exception To be handled.
	 */
	public MWNumericArray getLabels() throws Exception {

		double[][] labelValues = new double[nLabels][nBags];

		for (int i = 0; i < nBags; i++) {
			MIMLBag bag = mimlDataSet.getBag(i);
			for (int k = 0; k < nLabels; k++) {
				if (bag.stringValue(labelIndices[k]).equals("1"))
					labelValues[k][i] = 1;
				else
					labelValues[k][i] = -1;
			}
		}

		MWNumericArray aQMDoubleArray = new MWNumericArray(labelValues, MWClassID.DOUBLE);

		return aQMDoubleArray;
	}

	/**
	 * Returns a MIMLBag in the format of a 1x1 MWCellArray in which the bag is
	 * stored in CellArray{1,1} as an nInstxnAttributes array of double.
	 * 
	 * @param bag A MIMLBag.
	 * @return Returns a MIMLBag in the format of a 1x1 MWCellArray in which the bag
	 *         is stored in CellArray{1,1} as an nInstxnAttributes array of double.
	 * @throws Exception To be handled.
	 */
	public MWCellArray getBagAsCell(MIMLBag bag) throws Exception {

		MWCellArray aCellArrayIn = new MWCellArray(1, 1);
		int nInst = bag.getNumInstances();

		double[][] values = new double[nInst][attributesPerBag];
		for (int j = 0; j < nInst; j++) {
			Instance instance = bag.getInstance(j);
			for (int k = 0; k < attributesPerBag; k++)
				values[j][k] = instance.value(k);
		}

		int indice[] = new int[2];
		indice[0] = 1;
		indice[1] = 1;
		aCellArrayIn.set(indice, values);

		return aCellArrayIn;
	}

	/**
	 * Returns a MIMLBag in the format of a 1x1 MWCellArray in which the bag is
	 * stored in CellArray{1,1} as an nInstxnAttributes array of double.
	 * 
	 * @param index The index of the bag in the MIMLInstances dataset.
	 * @return Returns a MIMLBag in the format of a 1x1 MWCellArray in which the bag
	 *         is stored in CellArray{1,1} as an nInstxnAttributes array of double.
	 * @throws Exception To be handled.
	 */
	public MWCellArray getBagAsCell(int index) throws Exception {

		MIMLBag bag = mimlDataSet.getBag(index);
		return getBagAsCell(bag);
	}

	/**
	 * Returns a MIMLBag in the format of a nInstxnAttributes MWNumericArray of
	 * double.
	 * 
	 * @param bag A MIMLBag
	 * @return Returns a MIMLBag in the format of a nInstxnAttributes MWNumericArray
	 *         of double.
	 * @throws Exception To be handled.
	 */
	public MWNumericArray getBagAsArray(MIMLBag bag) throws Exception {
		int nInst = bag.getNumInstances();
		double[][] values = new double[nInst][attributesPerBag];
		for (int j = 0; j < nInst; j++) {
			Instance instance = bag.getInstance(j);
			for (int k = 0; k < attributesPerBag; k++)
				values[j][k] = instance.value(k);

		}
		MWNumericArray aDoubleArray = new MWNumericArray(values, MWClassID.DOUBLE);
		return aDoubleArray;
	}

	/**
	 * Returns a bag in the format of a nInstxnAttributes array of double.
	 * 
	 * @param index The index of the bag in the MIMLInstances dataset.
	 * @return A MIMLBag
	 * @throws Exception To be handled.
	 */
	public MWNumericArray getBagAsArray(int index) throws Exception {

		MIMLBag bag = mimlDataSet.getBag(index);
		return getBagAsArray(bag);
	}

	/**
	 * Returns label associations of a MIMLbag in the format of a nLabelsx1
	 * MWNumericArray of double. If the bag belongs to the jth label, then
	 * aDoubleArray(j,1) equals +1, otherwise aDoubleArray(j,1) equals -1.
	 *
	 * @param bag A MIMLBag.
	 * @return label associations of a bag in the format of a nLabelsx1
	 *         MWNumericArray of double.
	 * @throws Exception To be handled.
	 */
	public MWNumericArray getLabels(MIMLBag bag) throws Exception {

		double[][] labelValues = new double[nLabels][1];

		for (int k = 0; k < nLabels; k++) {
			if (bag.stringValue(labelIndices[k]).equals("1"))
				labelValues[k][0] = 1;
			else
				labelValues[k][0] = -1;

		}
		MWNumericArray aQMDoubleArray = new MWNumericArray(labelValues, MWClassID.DOUBLE);
		System.out.println(aQMDoubleArray.toString());

		return aQMDoubleArray;
	}

	/**
	 * Returns label associations of a MIMLbag in the format of a nLabelsx1
	 * MWNumericArray of double. If the bag belongs to the jth label, then
	 * aDoubleArray(j) equals +1, otherwise aDoubleArray(j,1) equals -1.
	 * 
	 * @param index The index of the bag in the MIMLInstances dataset.
	 * @return label associations of a bag in the format of a nLabelsx1
	 *         MWNumericArray of double.
	 * @throws Exception To be handled.
	 */
	public MWNumericArray getLabels(int index) throws Exception {

		MIMLBag bag = mimlDataSet.getBag(index);
		return getLabels(bag);
	}

}
