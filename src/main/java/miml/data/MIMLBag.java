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

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * 
 * Class inheriting from DenseInstance to represent a MIML bag.
 * 
 * @author Ana I. Reyes Melero
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20190315
 *
 */

public class MIMLBag extends DenseInstance implements Instance {

	/** Generated Serial version UID. */
	private static final long serialVersionUID = 1L;

	/**
	 * Constructor.
	 * 
	 * @param instance
	 *            A Weka's Instance to be transformed into a Bag.
	 * @throws Exception
	 *             To be handled in an upper level.
	 * 
	 */
	public MIMLBag(Instance instance) throws Exception {
		super(instance);
		m_AttValues = instance.toDoubleArray();
		m_Weight = instance.weight();
		m_Dataset = instance.dataset();
	}

	/**
	 * Returns an instance of the Bag with index bagIndex.
	 * 
	 * @param bagIndex
	 *            The index number.
	 * @return Instance.
	 * 
	 */
	public Instance getInstance(int bagIndex) {
		return this.relationalValue(1).instance(bagIndex);
	}

	/**
	 * Gets the total number of attributes of the Bag. This number includes
	 * attributes corresponding to labels. Instead the relational attribute, the
	 * number of attributes contained in the relational attribute is considered. For
	 * instance, in the relation above, the output of the method is 8.<br>
	 * 
	 * &#064;relation toy<br>
	 * &#064;attribute id {bag1,bag2}<br>
	 * &#064;attribute bag relational<br>
	 * &#064;attribute f1 numeric<br>
	 * &#064;attribute f2 numeric<br>
	 * &#064;attribute f3 numeric<br>
	 * &#064;end bag<br>
	 * &#064;attribute label1 {0,1}<br>
	 * &#064;attribute label2 {0,1}<br>
	 * &#064;attribute label3 {0,1}<br>
	 * &#064;attribute label4 {0,1}<br>
	 * 
	 * @return Total number of attributes of the Bag.
	 */
	public int getNumAttributesWithRelational() {
		return this.numAttributes() + this.relationalValue(1).numAttributes() - 1;
	}

	/**
	 * Gets the number of attributes of in the relational attribute of a Bag. For
	 * instance, in the relation above, the output of the method is 3.<br>
	 * 
	 * &#064;relation toy<br>
	 * &#064;attribute id {bag1,bag2}<br>
	 * &#064;attribute bag relational<br>
	 * &#064;attribute f1 numeric<br>
	 * &#064;attribute f2 numeric<br>
	 * &#064;attribute f3 numeric<br>
	 * &#064;end bag<br>
	 * &#064;attribute label1 {0,1}<br>
	 * &#064;attribute label2 {0,1}<br>
	 * &#064;attribute label3 {0,1}<br>
	 * &#064;attribute label4 {0,1}<br>
	 * 
	 * @return The number of attributes.
	 */
	public int getNumAttributesInABag() {
		return this.relationalValue(1).numAttributes();
	}

	/**
	 * Gets the number of instances of the Bag.
	 * 
	 * @return The number of instances of the Bag.
	 */
	public int getNumInstances() {
		return this.relationalValue(1).numInstances();
	}

	/**
	 * Gets a bag in the form of a set of instances considering just the relational
	 * information. Neither the identifier attribute of the Bag nor label attributes
	 * are included. For instance, given the relation toy above, the output of the
	 * method is the relation bag.<br>
	 * 
	 * &#064;relation toy<br>
	 * &#064;attribute id {bag1,bag2}<br>
	 * &#064;attribute bag relational<br>
	 * &#064;attribute f1 numeric<br>
	 * &#064;attribute f2 numeric<br>
	 * &#064;attribute f3 numeric<br>
	 * &#064;end bag<br>
	 * &#064;attribute label1 {0,1}<br>
	 * &#064;attribute label2 {0,1}<br>
	 * &#064;attribute label3 {0,1}<br>
	 * &#064;attribute label4 {0,1}<br>
	 * 
	 * &#064;relation bag<br>
	 * &#064;attribute f1 numeric<br>
	 * &#064;attribute f2 numeric<br>
	 * &#064;attribute f3 numeric<br>
	 * 
	 * @return Instances.
	 * @throws Exception
	 *             To be handled in an upper level.
	 */
	public Instances getBagAsInstances() throws Exception {
		Instances bags = this.relationalValue(1);
		return bags;
	}

	/**
	 * Sets the value of attrIndex attribute of the instanceIndex to a certain value.
	 * 
	 * @param instanceIndex
	 * 		The index of the instance.
	 * @param attrIndex
	 * 		The index of the attribute.
	 * @param value
	 * 		The value to be set.
	 */
	public void setValue(int instanceIndex, int attrIndex, double value)
	{
		this.getInstance(instanceIndex).setValue(attrIndex, value);		
	}
}
