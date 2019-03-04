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

package es.uco.kdis.miml.core.distance;

import java.io.Serializable;

import es.uco.kdis.miml.data.MIMLBag;
import weka.core.Instances;

/**
 * Interface to implements the metrics used to measure the distance between bags
 * of a data sets.
 *
 * @author Alvaro A. Belmonte
 * @author Amelia Zafra
 * @author Eva Gigaja
 * @version 20180619
 */
public interface IDistance extends Serializable {

	/**
	 * Get the distance between two bags.
	 *
	 * @param first  first Bag
	 * @param second second Bag
	 * @return distance between two bags
	 * @throws Exception the exception
	 */
	public double distance(MIMLBag first, MIMLBag second) throws Exception;

	/**
	 * Get the distance between two bags in the form of a set of instances.
	 *
	 * @param first  first Bag as Instances
	 * @param second second Bag as Instances
	 * @return distance between two bags
	 * @throws Exception the exception
	 */
	public double distance(Instances first, Instances second) throws Exception;

}
