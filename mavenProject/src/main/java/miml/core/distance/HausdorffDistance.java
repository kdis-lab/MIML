package miml.core.distance;

import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;

public abstract class HausdorffDistance implements IDistance {

	/**  */
	private static final long serialVersionUID = -1709140241905927188L;
	DistanceFunction dfun = null;
	Instances dataSet = null;

	public HausdorffDistance() {
		this.dfun = new EuclideanDistance();
	}

	public HausdorffDistance(MIMLInstances bags) throws Exception {
		setInstances(bags);
	}

	public boolean hasInstances() {
		return (dfun.getInstances() != null);
	}

	@Override
	public double distance(MIMLBag first, MIMLBag second) throws Exception {

		Instances firstDataset = first.getBagAsInstances();
		Instances secondDataset = second.getBagAsInstances();

		return (distance(firstDataset, secondDataset));
	}

	@Override
	public double distance(Instance bag1, Instance bag2) throws Exception {
		Instances relational1 = bag1.relationalValue(1);
		Instances relational2 = bag2.relationalValue(1);

		return distance(relational1, relational2);
	}

	@Override
	public void setInstances(MIMLInstances bags) throws Exception {
		if (bags.getNumBags() < 1)
			throw new Exception("To compute distance at least one bag is needed to initalize data set");

		dataSet = new Instances(bags.getBagAsInstances(0));
		for (int i = 1; i < bags.getNumBags(); i++) {
			Instances bag_i = bags.getBag(i).getBagAsInstances();
			dataSet.addAll(bag_i);
		}

		this.dfun = new EuclideanDistance(dataSet);
	}

	@Override
	public void setInstances(Instances bags) throws Exception {
		if (bags.numInstances() < 1)
			throw new Exception("To compute distance at least one bag is needed to initalize data set");

		dataSet = new Instances(bags.instance(0).relationalValue(1));
		for (int i = 1; i < bags.numInstances(); i++) {
			Instances bag_i = bags.instance(i).relationalValue(1);
			dataSet.addAll(bag_i);
		}

		this.dfun = new EuclideanDistance(dataSet);
	}

	@Override
	public void update(MIMLBag bag) throws Exception {
		Instances relational = bag.getBagAsInstances();
		for (int i = 0; i < relational.numInstances(); i++) {
			this.dfun.update(relational.get(i));
		}
	}

	@Override
	public void update(Instance bag) throws Exception {
		Instances relational = bag.relationalValue(1);
		;
		for (int i = 0; i < relational.numInstances(); i++) {
			this.dfun.update(relational.get(i));
		}
	}

}
