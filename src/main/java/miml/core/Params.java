package miml.core;

/**
 * This class contains the list of classes and objects needed to create a new
 * instance of a Multi Label classifier through a specific constructor.
 *
 * @author Aurora Esteban Toscano
 * @author Álvaro A. Belmonte
 * @author Eva Gibaja
 * @author Amelia Zafra
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