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

package miml.core.distance;

import java.io.Serializable;

import miml.data.MIMLBag;
import weka.core.Instances;

/**
 * Interface to implements the metrics used to measure the distance between {@link MIMLBag}
 * of a data sets.
 *
 * @author Alvaro A. Belmonte
 * @author Amelia Zafra
 * @author Eva Gigaja
 * @version 20180619
 */
public interface IDistance extends Serializable {

	/**
	 * Get the distance between two {@link MIMLBag}.
	 *
	 * @param first  First bag.
	 * @param second Second bag.
	 * @return Distance between two bags.
	 * @throws Exception if occurred an error during distance calculation,
	 */
	public double distance(MIMLBag first, MIMLBag second) throws Exception;

	/**
	 * Get the distance between two bags in the form of a set of {@link Instances}.
	 *
	 * @param first  First bag as instances.
	 * @param second Second Bag as Instances.
	 * @return Distance between two bags.
	 * @throws Exception if occurred an error during distance calculation.
	 */
	public double distance(Instances first, Instances second) throws Exception;

}
