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

package miml.classifiers.miml;

import java.io.Serializable;

import miml.data.MIMLInstances;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import weka.core.Instance;

/**
 * Common interface for MIML classifiers.
 *
 * @author Alvaro A. Belmonte
 * @author Amelia Zafra
 * @author Eva Gigaja
 * @version 20180619
 */
public interface IMIMLClassifier extends MultiLabelLearner, Serializable {

	/*
	 * (non-Javadoc)
	 * 
	 * @see mulan.classifier.MultiLabelLearner#makePrediction(weka.core.Instance)
	 */
	@Override
	public MultiLabelOutput makePrediction(Instance instance) throws Exception;

	/**
	 * Builds the learner model from specified {@link MIMLInstances} data.
	 *
	 * @param trainingSet Set of training data, upon which the learner model should be
	 *                  built.
	 * @throws Exception If learner model was not created successfully.
	 */
	public void build(MIMLInstances trainingSet) throws Exception;

	/*
	 * (non-Javadoc)
	 * 
	 * @see mulan.classifier.MultiLabelLearner#setDebug(boolean)
	 */
	@Override
	public void setDebug(boolean debug);

	/*
	 * (non-Javadoc)
	 * 
	 * @see mulan.classifier.MultiLabelLearner#makeCopy()
	 */
	@Override
	public IMIMLClassifier makeCopy() throws Exception;

}
