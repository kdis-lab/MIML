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
package miml.classifiers.mi;

import weka.classifiers.mi.MISMO;
import weka.core.Instance;

/**
 * 
 * Wrapper for MISMO algorithm to work in MIML to MI classifiers.
 * 
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @author Alvaro A. Belmonte
 * @version 20190525
 *
 */
public class MISMOWrapper extends MISMO {

	/** Generated Serial version UID. */
	private static final long serialVersionUID = 1L;

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.classifiers.mi.MISMO#MISMO()
	 */
	public MISMOWrapper() {
		super();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.classifiers.mi.MISMO#distributionForInstance(weka.core.Instance)
	 */
	@Override
	public double[] distributionForInstance(Instance inst) throws Exception {

		// Before prediction Mulan sets the class value to '?' (missing), before calling
		// MISMO, this value is set to '0' (not predicted)

		inst.setValue(2, 0);
		return super.distributionForInstance(inst);

	}
}
