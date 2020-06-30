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

import java.io.File;

import org.apache.commons.configuration2.Configuration;
import org.apache.commons.configuration2.XMLConfiguration;
import org.apache.commons.configuration2.builder.FileBasedConfigurationBuilder;
import org.apache.commons.configuration2.builder.fluent.Parameters;
import org.apache.commons.configuration2.builder.fluent.XMLBuilderParameters;
import org.apache.commons.configuration2.ex.ConfigurationException;

import miml.classifiers.miml.IMIMLClassifier;
import miml.classifiers.miml.mimlTOmi.MIMLClassifierToMI;
import miml.classifiers.miml.mimlTOml.MIMLClassifierToML;
import miml.evaluation.IEvaluator;
import miml.report.IReport;

/**
 * Class used to read a xml file and configure an experiment.
 * 
 * @author Alvaro A. Belmonte
 * @author Amelia Zafra
 * @author Eva Gigaja
 * @version 20180619
 */
public class ConfigLoader {

	/** Configuration object. */
	protected Configuration configuration;

	/**
	 * Gets the experiment's configuration.
	 *
	 * @return The configuration used during experimentation.
	 */
	public Configuration getConfiguration() {
		return configuration;
	}

	/**
	 * Sets the configuration for the experiment.
	 *
	 * @param configuration A new configuration.
	 */
	public void setConfiguration(Configuration configuration) {
		this.configuration = configuration;
	}

	/**
	 * Constructor that sets the configuration file
	 *
	 * @param path The path of config file.
	 * @throws ConfigurationException if occurred an error during the loading of the configuration.
	 */
	public ConfigLoader(String path) throws ConfigurationException {
		
		Parameters params = new Parameters();

		XMLBuilderParameters px = params.xml();

		FileBasedConfigurationBuilder<XMLConfiguration> builder = new FileBasedConfigurationBuilder<XMLConfiguration>(
				XMLConfiguration.class);

		builder.configure(px.setFileName(path));

		try {
			configuration = builder.getConfiguration();
		} catch (ConfigurationException e) {
			e.printStackTrace();
		}

		ConfigParameters.setConfigFileName(new File(path).getName());

	}

	/**
	 * Read current configuration to load and configure the classifier.
	 *
	 * @return A MIMLClassifier.
	 * @throws Exception if the classifier couldn't be loaded correctly.
	 */
	@SuppressWarnings("unchecked")
	public IMIMLClassifier loadClassifier() throws Exception {

		IMIMLClassifier classifier = null;

		String clsName = configuration.getString("classifier[@name]");
		// Instantiate the classifier class used in the experiment
		Class<? extends IMIMLClassifier> clsClass = (Class<? extends IMIMLClassifier>) Class.forName(clsName);

		classifier = clsClass.newInstance();
		// Configure the classifier
		if (classifier instanceof IMIMLClassifier)
			((IConfiguration) classifier).configure(configuration.subset("classifier"));
						

		ConfigParameters.setAlgorithmName(classifier.getClass().getSimpleName());

		return classifier;
	}

	/**
	 * Read current configuration to load and configure the evaluator.
	 *
	 * @return A evaluator for MIML Classifiers.
	 * @throws Exception if the class loaded can't be loaded.
	 */
	@SuppressWarnings({ "rawtypes", "unchecked" })
	public IEvaluator loadEvaluator() throws Exception {

		IEvaluator evaluator = null;

		String evalName = configuration.getString("evaluator[@name]");
		// Instantiate the evaluator class used in the experiment
		Class<? extends IEvaluator> evalClass = (Class<? extends IEvaluator>) Class.forName(evalName);

		evaluator = evalClass.newInstance();
		// Configure the evaluator
		if (evaluator instanceof IEvaluator)
			((IConfiguration) evaluator).configure(configuration.subset("evaluator"));

		return evaluator;
	}

	/**
	 * Read current configuration to load and configure the report.
	 *
	 * @return the MIML report
	 * @throws Exception if the class can't be loaded.
	 */
	@SuppressWarnings("unchecked")
	public IReport loadReport() throws Exception {

		IReport report = null;

		String reportName = configuration.getString("report[@name]");

		// Instantiate the report class used in the experiment
		Class<? extends IReport> clsClass = (Class<? extends IReport>) Class.forName(reportName);

		report = clsClass.newInstance();

		// Configure the report
		if (report instanceof IReport)
			((IConfiguration) report).configure(configuration.subset("report"));

		return report;
	}

}
