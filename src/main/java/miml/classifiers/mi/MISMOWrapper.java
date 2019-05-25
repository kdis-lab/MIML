package miml.classifiers.mi;

import weka.classifiers.mi.MISMO;
import weka.core.Instance;

public class MISMOWrapper extends MISMO {

	private static final long serialVersionUID = 1L;

	public MISMOWrapper() {
		super();
	}

	@Override
	public double[] distributionForInstance(Instance inst) throws Exception {

		// Before prediction Mulan sets the class value to '?' (missing), before calling
		// MISMO, this value is set to '0' (not predicted)

		inst.setValue(2, 0);
		return super.distributionForInstance(inst);

	}
}
