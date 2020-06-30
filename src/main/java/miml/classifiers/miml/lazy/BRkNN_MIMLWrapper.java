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
package miml.classifiers.miml.lazy;

import org.apache.commons.configuration2.Configuration;

import mulan.classifier.lazy.BRkNN;
import mulan.classifier.lazy.BRkNN.ExtensionType;

/**
 * Wrapper for BRkNN of Mulan Library. BRkNN is the simple BR implementation of
 * the KNN algorithm. For more information, see<br>
 * <p>
 * Eleftherios Spyromitros, Grigorios Tsoumakas, Ioannis Vlahavas: An Empirical
 * Study of Lazy Multilabel Classification Algorithms. In: Proc. 5th Hellenic
 * Conference on Artificial Intelligence (SETN 2008), 2008.
 * </p>
 */
public class BRkNN_MIMLWrapper extends MultiLabelKNN_MIMLWrapper {


	/**
	 * Generated Serial version UID.
	 */

	private static final long serialVersionUID = 1L;
	/**
	 * The type of extension to be used:
	 * <ul>
	 * <li>NONE: Standard BR.</li>
	 * <li>EXTA: Predict top ranked label in case of empty prediction set.</li>
	 * <li>EXTB: Predict top n ranked labels based on size of labelset in
	 * neighbours.</li>
	 * </ul>
	 */
	private ExtensionType extension = ExtensionType.NONE;

	/**
	 * No-arg constructor for xml configuration
	 */
	public BRkNN_MIMLWrapper() {
	}

	/**
	 * Default constructor.
	 * 
	 * @param metric
	 *            The distance metric between bags considered by the classifier.
	 */

	public BRkNN_MIMLWrapper(DistanceFunction_MIMLWrapper metric) {
		super(metric, 10);
		this.classifier = new BRkNN(10);
	}

	/**
	 * A constructor that sets the number of neighbours.
	 *
	 * @param metric
	 *            The distance metric between bags considered by the classifier.
	 * @param numOfNeighbours
	 *            the number of neighbours.
	 */
	public BRkNN_MIMLWrapper(DistanceFunction_MIMLWrapper metric, int numOfNeighbours) {
		super(metric, numOfNeighbours);		
		this.classifier = new BRkNN(numOfNeighbours);
	}

	/**
	 * Constructor giving the option to select an extension of the base version.
	 *
	 * @param metric
	 *            The distance metric between bags considered by the classifier.
	 * @param numOfNeighbours
	 *            the number of neighbours
	 * @param ext
	 *            the extension to use (see {@link ExtensionType}).
	 */
	public BRkNN_MIMLWrapper(DistanceFunction_MIMLWrapper metric, int numOfNeighbours, ExtensionType ext) {
		super(metric, numOfNeighbours);		
		this.extension = ext;
		this.classifier = new BRkNN(numOfNeighbours, ext);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see miml.core.IConfiguration#configure(org.apache.commons.configuration2.
	 * Configuration)
	 */
	@Override
	public void configure(Configuration configuration) {

		super.configure(configuration);
		String ext = configuration.getString("extension");
		if (ext.equalsIgnoreCase("EXTA"))
			this.extension = ExtensionType.EXTA;
		else if (ext.equalsIgnoreCase("EXTB"))
			this.extension = ExtensionType.EXTB;
		else
			this.extension = ExtensionType.NONE;
		this.classifier = new BRkNN(numOfNeighbours, extension);
	}
	
	/***/
	/**
	 * Gets the type of extension to be used (see {@link ExtensionType}).
	 * @return extension Extension to be used
	 */
	public ExtensionType getExtension() {
		return extension;
	}

	/**Sets the type of extension to be used (see {@link ExtensionType}).
	 * @param extension The new value of the type of extension.*/
	public void setExtension(ExtensionType extension) {
		this.extension = extension;
	}
}
