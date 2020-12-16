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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import mulan.data.LabelNodeImpl;
import mulan.data.LabelsBuilder;
import mulan.data.LabelsBuilderException;
import mulan.data.LabelsMetaDataImpl;
import mulan.data.MultiLabelInstances;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

/**
 * Class with methods to write to file a multi-label dataset. MIML format is
 * also supported.
 * 
 * @author F.J. Gonzalez
 * @author Eva Gibaja
 * @version 20161122
 */
public final class MLSave {

	private MLSave() {
	}
	
	/**
	 * Writes an arff file with a multi-label dataset. MIML format is also
	 * supported.
	 *
	 * @param instances A multi-label dataset.
	 * @param pathName  Name and path for file to write.
	 * @throws java.io.IOException To be handled in an upper level.
	 */
	public static void saveArff(MIMLInstances instances, String pathName) throws IOException {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances.getDataSet());
		saver.setFile(new File(pathName));
		saver.writeBatch();
		System.out.println("Arff dataset written to " + pathName);
	}

	/**
	 * Writes an arff file with a multi-label dataset. MIML format is also
	 * supported.
	 *
	 * @param instances A multi-label dataset.
	 * @param pathName  Name and path for file to write.
	 * @throws java.io.IOException To be handled in an upper level.
	 */
	public static void saveArff(MultiLabelInstances instances, String pathName) throws IOException {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances.getDataSet());
		saver.setFile(new File(pathName));
		saver.writeBatch();
		System.out.println("Arff dataset written to " + pathName);
	}

	/**
	 * Writes an arff file with an Instances dataset.
	 * 
	 *
	 * @param instances A dataset.
	 * @param pathName  Name and path for file to write.
	 * @throws java.io.IOException To be handled in an upper level.
	 */
	public static void saveArff(Instances instances, String pathName) throws IOException {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances);
		saver.setFile(new File(pathName));
		saver.writeBatch();
		System.out.println("Arff dataset written to " + pathName);
	}

	/**
	 * Writes an xml file with label definitions of a multi-label dataset. MIML
	 * format is also supported.
	 *
	 * @param instances A multi-label dataset.
	 * @param pathName  Name and path for file to write.
	 * @throws java.io.IOException To be handled in an upper level.
	 * @throws mulan.data.LabelsBuilderException To be handled in an upper level.
	 */
	public static void saveXml(MultiLabelInstances instances, String pathName)
			throws IOException, LabelsBuilderException {
		LabelsBuilder.dumpLabels(instances.getLabelsMetaData(), pathName);
		System.out.println("Xml file written to " + pathName);
	}

	/**
	 * Writes an xml file with label definitions of an instances dataset.
	 *
	 * @param instances A dataset.
	 * @param pathName  Name and path for file to write.
	 * @throws java.io.IOException To be handled in an upper level.
	 * @throws mulan.data.LabelsBuilderException To be handled in an upper level.
	 */
	public static void saveXml(Instances instances, String pathName) throws IOException, LabelsBuilderException {
		LabelsBuilder.createLabels(pathName);
		System.out.println("Xml file written to " + pathName);
	}

	/**
	 * Writes an xml file.
	 *
	 * @param labelNames An ArrayList&lt;String&gt; with label names.
	 * @param pathName   Name and path for file to write.
	 */
	public static void saveXml(ArrayList<String> labelNames, String pathName) {
		LabelsMetaDataImpl meta = new LabelsMetaDataImpl();
		for (int label = 0; label < labelNames.size(); label++) {
			meta.addRootNode(new LabelNodeImpl(labelNames.get(label)));
		}

		try {
			LabelsBuilder.dumpLabels(meta, pathName);
			System.out.println("Xml file written to " + pathName);
		} catch (LabelsBuilderException e) {
			File labelsFile = new File(pathName);
			if (labelsFile.exists()) {
				labelsFile.delete();
			}
			System.out.println("Construction of labels XML failed!");
		}
	}
}
