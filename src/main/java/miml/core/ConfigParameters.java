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
 * Class used to save config parameters to be used in reports.
 * 
 * @author Alvaro A. Belmonte
 * @author Amelia Zafra
 * @author Eva Gigaja
 * @version 20190524
 */
public final class ConfigParameters {
	

	/** The algorirthm used in the experimentation. */
	protected static String algorirthmName = "";
	
	/** The config filename used in the experimentation. */
	protected static String configFileName = "";
	
	/** The data filename used in the experimentation. */
	protected static String dataFileName = "";
	
	/** The classifier used in the experimentation. */
	protected static String classifierName = "";
	
	/** The transform method used in the experimentation. */
	protected static String transformMethod = "";
	
	protected static Boolean isDegenerative = false;

	/**
	 * Gets the algorirthm name.
	 *
	 * @return the algorirthm name
	 */
	public static String getAlgorirthmName() {
		return algorirthmName;
	}
	
	/**
	 * Sets the algorirthm name.
	 *
	 * @param algorirthmName the new algorirthm name
	 */
	public static void setAlgorirthmName(String algorirthmName) {
		ConfigParameters.algorirthmName = algorirthmName;
	}
	
	/**
	 * Gets the config file name.
	 *
	 * @return the config file name
	 */
	public static String getConfigFileName() {
		return configFileName;
	}
	
	/**
	 * Sets the config file name.
	 *
	 * @param configFileName the new config file name
	 */
	public static void setConfigFileName(String configFileName) {
		ConfigParameters.configFileName = configFileName;
	}
	
	/**
	 * Gets the data file name.
	 *
	 * @return the data file name
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
	 * Gets the classifier name
	 * 
	 * @return the classifier name
	 */
	public static String getClassifierName() {
		return classifierName;
	}

	/**
	 * Sets the classifier name
	 * @param classifierName the classifier name
	 */
	public static void setClassifierName(String classifierName) {
		ConfigParameters.classifierName = classifierName;
	}

	/**
	 * Gets the transform method used in the experiment
	 * 
	 * @return the transform method used in the experiment
	 */
	public static String getTransformMethod() {
		return transformMethod;
	}

	/**
	 * Sets the transform method used in the experiment
	 * 
	 * @param transformMethod the transform method used in the experiment
	 */
	public static void setTransformMethod(String transformMethod) {
		ConfigParameters.transformMethod = transformMethod;
	}
	
	public static Boolean getIsDegenerative() {
		return isDegenerative;
	}

	public static void setIsDegenerative(Boolean isDegenerative) {
		ConfigParameters.isDegenerative = isDegenerative;
	}
	
	
	
}
