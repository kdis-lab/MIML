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

package miml.core;

/**
 * Class used to save configuration parameters to be used in reports.
 * 
 * @author Alvaro A. Belmonte
 * @author Amelia Zafra
 * @author Eva Gigaja
 * @version 20190524
 */
public final class ConfigParameters {

	/** The algorithm used in the experimentation. */
	protected static String algorithmName = "";

	/** The configuration filename used in the experimentation. */
	protected static String configFileName = "";

	/** The name of data file used in the experimentation. */
	protected static String dataFileName = "";

	/** The classifier used in the experimentation. */
	protected static String classifierName = "";

	/**
	 * The name of the method used in the experiment if this is a transformation
	 * method.
	 */
	protected static String transformationMethod = "";

	/**
	 * If the classifier configured in the experiment uses a method
	 * transformation.
	 */
	protected static Boolean isTransformation = false;

	/**
	 * Gets the algorithm name.
	 *
	 * @return The algorithm name.
	 */
	public static String getAlgorithmName() {
		return algorithmName;
	}

	/**
	 * Sets the algorithm name.
	 *
	 * @param algorithmName The new algorithm name.
	 */
	public static void setAlgorithmName(String algorithmName) {
		ConfigParameters.algorithmName = algorithmName;
	}

	/**
	 * Gets the configuration file name.
	 *
	 * @return The configuration file name.
	 */
	public static String getConfigFileName() {
		return configFileName;
	}

	/**
	 * Sets the configuration file name.
	 *
	 * @param configFileName The new configuration file name.
	 */
	public static void setConfigFileName(String configFileName) {
		ConfigParameters.configFileName = configFileName;
	}

	/**
	 * Gets the name of data file.
	 *
	 * @return The name of data file.
	 */
	public static String getDataFileName() {
		return dataFileName;
	}

	/**
	 * Sets the data file name.
	 *
	 * @param dataFileName the new data file name
	 */
	public static void setDataFileName(String dataFileName) {
		ConfigParameters.dataFileName = dataFileName;
	}

	/**
	 * Gets the classifier name.
	 * 
	 * @return The classifier name.
	 */
	public static String getClassifierName() {
		return classifierName;
	}

	/**
	 * Sets the classifier name.
	 * 
	 * @param classifierName The classifier name.
	 */
	public static void setClassifierName(String classifierName) {
		ConfigParameters.classifierName = classifierName;
	}

	/**
	 * Gets the transformation method used in the experiment.
	 * 
	 * @return The transformation method used in the experiment.
	 */
	public static String getTransformationMethod() {
		return transformationMethod;
	}

	/**
	 * Sets the transformation method used in the experiment.
	 * 
	 * @param transformationMethod The transformation method used in the experiment.
	 */
	public static void setTransformationMethod(String transformationMethod) {
		ConfigParameters.transformationMethod = transformationMethod;
	}

	/**
	 * Gets if the method used is transformation.
	 * 
	 * @return True if the method used is transformation.
	 */
	public static Boolean getIsTransformation() {
		return isTransformation;
	}

	/**
	 * Sets if the method used is transformation.
	 * 
	 * @param isTransformation If the method used is transformation.
	 */
	public static void setIsTransformation(Boolean isTransformation) {
		ConfigParameters.isTransformation = isTransformation;
	}

}
