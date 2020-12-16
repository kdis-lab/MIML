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

package miml.core;

/**
 * This class contains the list of classes and objects needed to create a new
 * instance of a Multi Label classifier through a specific constructor.
 *
 * @author Aurora Esteban Toscano
 * @author Alvaro A. Belmonte
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20200626
 **/
public class Params {

	/**
	 * List of classes needed by the Multi Label classifier's constructor.
	 */
	private Class<?>[] classes;

	/**
	 * List of the values for the classes array
	 */
	private Object[] objects;

	/**
	 * Generic constructor
	 * 
	 * @param classes The list of classes needed by the Multi Label classifier's
	 *                constructor.
	 * @param objects The list of the values for the classes array.
	 */
	public Params(Class<?>[] classes, Object[] objects) {
		this.classes = classes;
		this.objects = objects;
	}

	/**
	 * @return the classes
	 */
	public Class<?>[] getClasses() {
		return classes;
	}

	/**
	 * @param classes the classes to set
	 */
	public void setClasses(Class<?>[] classes) {
		this.classes = classes;
	}

	/**
	 * @return the objects
	 */
	public Object[] getObjects() {
		return objects;
	}

	/**
	 * @param objects the objects to set
	 */
	public void setObjects(Object[] objects) {
		this.objects = objects;
	}
	
	
}