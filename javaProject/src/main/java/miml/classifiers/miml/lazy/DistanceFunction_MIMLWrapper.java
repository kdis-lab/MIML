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

import java.io.Serializable;
import java.util.Enumeration;

import miml.core.distance.IDistance;
import miml.data.MIMLBag;
import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

public class DistanceFunction_MIMLWrapper implements Serializable, DistanceFunction {

	/**
	 * Wrapper for using IDistance metrics of MIML package with Mulan Lazy
	 * algorithms.
	 */
	private static final long serialVersionUID = 1111038321156350822L;

	/** Metric to measure distance between bags. */
	IDistance metric;

	/**
	 * Sets the metric to be used.
	 * 
	 * @param metric The metric to be used.
	 */
	public void setMetric(IDistance metric) {
		this.metric = metric;
	}

	/**
	 * Constructor that sets the metric to be used.
	 * 
	 * @param metric The metric to be used.
	 */
	public DistanceFunction_MIMLWrapper(IDistance metric) {
		super();
		this.metric = metric;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.core.DistanceFunction#distance(weka.core.Instance,
	 * weka.core.Instance, double, weka.core.neighboursearch.PerformanceStats)
	 */
	@Override
	public double distance(Instance arg0, Instance arg1, double arg2, PerformanceStats arg3) {

		double finalDistance = 0.0;

		try {

			MIMLBag first = new MIMLBag(arg0);
			MIMLBag second = new MIMLBag(arg1);
			finalDistance = metric.distance(first, second);

		} catch (Exception e) {
			e.printStackTrace();
		}

		return finalDistance;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.core.OptionHandler#getOptions()
	 */
	@Override
	public String[] getOptions() {
		// TODO Auto-generated method stub
		return null;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.core.OptionHandler#setOptions(java.lang.String[])
	 */
	@Override
	public void setOptions(String[] arg0) throws Exception {
		// TODO Auto-generated method stub

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.core.DistanceFunction#distance(weka.core.Instance,
	 * weka.core.Instance)
	 */
	@Override
	public double distance(Instance arg0, Instance arg1) {
		return distance(arg0, arg1, Double.POSITIVE_INFINITY);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.core.DistanceFunction#distance(weka.core.Instance,
	 * weka.core.Instance, weka.core.neighboursearch.PerformanceStats)
	 */
	@Override
	public double distance(Instance arg0, Instance arg1, PerformanceStats arg2) throws Exception {
		return distance(arg0, arg1, Double.POSITIVE_INFINITY, arg2);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.core.DistanceFunction#distance(weka.core.Instance,
	 * weka.core.Instance, double)
	 */
	@Override
	public double distance(Instance arg0, Instance arg1, double arg2) {
		return distance(arg0, arg1, arg2, null);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.core.DistanceFunction#setInstances(weka.core.Instances)
	 */
	@Override
	public void setInstances(Instances insts) {
		// TODO Auto-generated method stub

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.core.DistanceFunction#getInstances()
	 */
	@Override
	public Instances getInstances() {
		// TODO Auto-generated method stub
		return null;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.core.DistanceFunction#setAttributeIndices(java.lang.String)
	 */
	@Override
	public void setAttributeIndices(String value) {
		// TODO Auto-generated method stub

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.core.DistanceFunction#getAttributeIndices()
	 */
	@Override
	public String getAttributeIndices() {
		// TODO Auto-generated method stub
		return null;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.core.DistanceFunction#setInvertSelection(boolean)
	 */
	@Override
	public void setInvertSelection(boolean value) {
		// TODO Auto-generated method stub

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.core.DistanceFunction#getInvertSelection()
	 */
	@Override
	public boolean getInvertSelection() {
		// TODO Auto-generated method stub
		return false;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.core.DistanceFunction#postProcessDistances(double[])
	 */
	@Override
	public void postProcessDistances(double[] distances) {
		// TODO Auto-generated method stub

	}

	/*
	 * @see weka.core.DistanceFunction#update(weka.core.Instance)
	 */
	@Override
	public void update(Instance ins) {
		// TODO Auto-generated method stub

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see weka.core.DistanceFunction#listOptions()
	 */
	@Override
	public Enumeration<?> listOptions() {
		// TODO Auto-generated method stub
		return null;
	}

}
