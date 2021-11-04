package miml.core.distance;

import miml.data.MIMLBag;
import miml.data.MIMLInstances;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instances;

public abstract class HausdorffDistance implements IDistance {

	/**  */
	private static final long serialVersionUID = -1709140241905927188L;
	DistanceFunction dfun = null;	
	Instances dataSet = null;
	
	public HausdorffDistance() {
		this.dfun = new EuclideanDistance();
	}

	public HausdorffDistance(MIMLInstances bags) throws Exception
	{setInstances(bags);}
	
	public boolean hasInstances() {
		return (dfun.getInstances()!= null);		
	}
	
	/*
	 * (non-Javadoc)
	 * 
	 * @see core.distance.IDistance#distance(data.Bag, data.Bag)
	 */
	@Override
	public double distance(MIMLBag first, MIMLBag second) throws Exception {

		Instances firstDataset = first.getBagAsInstances();
		Instances secondDataset = second.getBagAsInstances();

		return (distance(firstDataset, secondDataset));
	}
	
	public void setInstances(MIMLInstances bags) throws Exception {
		if(bags.getNumBags()<1)
			throw new Exception("To compute distance at least one bag is needed to initalize data set");		
		
		dataSet = new Instances(bags.getBagAsInstances(0));
		for(int i=1; i<bags.getNumBags(); i++)
		{
			Instances bag_i= bags.getBag(i).getBagAsInstances();
			dataSet.addAll(bag_i);
		}	
			
		this.dfun = new EuclideanDistance(dataSet);		
	}
	
	public void update(MIMLBag bag) throws Exception
	{
		Instances relational = bag.getBagAsInstances();
		for(int i=0; i<relational.numInstances(); i++)
		{
			this.dfun.update(relational.get(i));
		}		
	}
}
