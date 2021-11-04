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
package miml.classifiers.miml.optimizer;

import org.apache.commons.configuration2.Configuration;

import miml.classifiers.miml.MIMLClassifier;
import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import miml.data.MWWrapper;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelOutput;

/**
 * <p>
 * Class to execute the <a href="http://www.lamda.nju.edu.cn/code_KISAR.ashx">KISAR
 * algorithm</a> for MIML data. For more information, see <em>Li, Y. F., Hu, J.
 * H., Jiang, Y., &amp; Zhou, Z. H. (2012, July). Towards discovering what patterns
 * trigger what labels. In Twenty-Sixth AAAI Conference on Artificial
 * Intelligence.</em>.
 * </p>
 * 
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20211031
 */
public class Kisar_MIMLWrapper extends MIMLClassifier {

	/** For serialization */
	private static final long serialVersionUID = 2035495789185416878L;
	
	/** Wrapper for Matlab data types. */
	protected MWWrapper wrapper;
	
	public Kisar_MIMLWrapper() {
		// TODO Auto-generated constructor stub
	}

	@Override
	public void configure(Configuration configuration) {
		// TODO Auto-generated method stub

	}

	@Override
	protected void buildInternal(MIMLInstances trainingSet) throws Exception {
		// TODO Auto-generated method stub

	}

	@Override
	protected MultiLabelOutput makePredictionInternal(MIMLBag instance) throws Exception, InvalidDataException {
		// TODO Auto-generated method stub
		return null;
	}

}
