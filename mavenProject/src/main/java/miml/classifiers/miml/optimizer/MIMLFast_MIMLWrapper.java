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
 * Class to execute the
 * <a href="http://www.lamda.nju.edu.cn/code_MIMLfast.ashx">MIMLFast
 * algorithm</a> for MIML data. For more information, see <em>ZHuang, S. J.,
 * Gao, W., &amp; Zhou, Z. H. (2018). Fast multi-instance multi-label learning. IEEE
 * transactions on pattern analysis and machine intelligence, 41(11),
 * 2614-2627.</em>.
 * </p>
 * 
 * @author Eva Gibaja
 * @author Amelia Zafra
 * @version 20211031
 */
public class MIMLFast_MIMLWrapper extends MIMLClassifier {

	/** For serialization */
	private static final long serialVersionUID = -3196787945364929958L;

	/** Wrapper for Matlab data types. */
	protected MWWrapper wrapper;
	
	public MIMLFast_MIMLWrapper() {
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
