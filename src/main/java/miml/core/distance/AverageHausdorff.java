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

import java.util.Arrays;
import java.util.stream.DoubleStream;

import miml.data.MIMLBag;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class that implements Average Hausdorff metric to measure the distance
 * between 2 bags of a data set.
 *
 * @author Alvaro A. Belmonte
 * @author Amelia Zafra
 * @author Eva Gigaja
 * @version 20180619
 */
public class AverageHausdorff implements IDistance {

	/** Generated Serial version UID. */
	private static final long serialVersionUID = -2002702276955682922L;

	/*
	 * (non-Javadoc)
	 * 
	 * @see core.distance.IDistance#distance(data.Bag, data.Bag)
	 */
	@Override
	public double distance(MIMLBag first, MIMLBag second) throws Exception {

		EuclideanDistance euclideanDistance = new EuclideanDistance(first.getBagAsInstances());

		int nInstances = second.getBagAsInstances().size();

		int idx = 0;
		double sumU = 0.0;
		double[] minDistancesV = new double[nInstances];
		Arrays.fill(minDistancesV, Double.MAX_VALUE);

		for (Instance u : first.getBagAsInstances()) {

			double minDistance = Double.MAX_VALUE;

			for (Instance v : second.getBagAsInstances()) {

				double distance = euclideanDistance.distance(u, v);

				if (distance < minDistance)
					minDistance = distance;

				if (distance < minDistancesV[idx])
					minDistancesV[idx] = distance;

				idx++;
			}

			idx = 0;
			sumU += minDistance;
		}

		double sumV = DoubleStream.of(minDistancesV).sum();

		return (sumU + sumV) / (first.getNumInstances() + second.getNumInstances());
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see core.distance.IDistance#distance(weka.core.Instances,
	 * weka.core.Instances)
	 */
	@Override
	public double distance(Instances first, Instances second) throws Exception {

		EuclideanDistance euclideanDistance = new EuclideanDistance(first);
		euclideanDistance.setDontNormalize(true);

		int nInstances = second.size();

		int idx = 0;
		double sumU = 0.0;
		double[] minDistancesV = new double[nInstances];
		Arrays.fill(minDistancesV, Double.MAX_VALUE);

		for (int i = 0; i < first.size(); ++i) {

			Instance u = first.instance(i);

			double minDistance = Double.MAX_VALUE;

			for (int j = 0; j < nInstances; ++j) {

				double distance = euclideanDistance.distance(u, second.instance(j));

				if (distance < minDistance)
					minDistance = distance;

				if (distance < minDistancesV[idx])
					minDistancesV[idx] = distance;

				idx++;
			}

			idx = 0;
			sumU += minDistance;
		}

		double sumV = DoubleStream.of(minDistancesV).sum();

		return (sumU + sumV) / (first.size() + second.size());
	}

}
