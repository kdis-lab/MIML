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

package es.uco.kdis.miml.tutorial;

import java.io.IOException;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import org.apache.commons.configuration2.ex.ConfigurationException;

import es.uco.kdis.miml.classifiers.miml.mimlTOml.MIMLClassifierML;
import es.uco.kdis.miml.core.ConfigLoader;
import es.uco.kdis.miml.evaluation.IEvaluator;
import es.uco.kdis.miml.transformation.mimlTOml.MIMLtoML;
import mulan.classifier.MultiLabelLearner;
import weka.core.Utils;

/**
 * Class that allow run any algorithm of the library configured by a file
 * configuration.
 * 
 * @author Alvaro A. Belmonte
 * @author Amelia Zafra
 * @author Eva Gigaja
 * @version 20180619
 */
public class PruebasML {

	public static String lastToken(String str, String separatorRegex) {
		String tokens[] = str.split(separatorRegex);
		return tokens[tokens.length - 1];
	}

	static Logger logger = Logger.getLogger("MyLog");
	static FileHandler fh;

	/**
	 * The main method to configure and run an algorithm.
	 *
	 * @param args the arguments(route of config file with the option -c)
	 * @throws Exception
	 * @throws ConfigurationException
	 */
	@SuppressWarnings({ "unused", "unchecked" })
	public static void main(String[] args) throws ConfigurationException, Exception {

		try {

			// This block configure the logger with handler and formatter
			fh = new FileHandler("logs/Pruebas_ML.log");
			logger.addHandler(fh);
			SimpleFormatter formatter = new SimpleFormatter();
			fh.setFormatter(formatter);

		} catch (SecurityException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		// Configuración base
		ConfigLoader loader = new ConfigLoader(Utils.getOption("c", args));

		String[] metodosTransformacion = new String[] {
				"es.uco.kdis.miml.transformation.mimlTOml.ArithmeticTransformation",
				"es.uco.kdis.miml.transformation.mimlTOml.GeometricTransformation",
				"es.uco.kdis.miml.transformation.mimlTOml.MiniMaxTransformation" };

		String[] clasificadoresBase = new String[] { "mulan.classifier.lazy.BRkNN", "mulan.classifier.lazy.DMLkNN",
				"mulan.classifier.lazy.IBLR_ML", "mulan.classifier.lazy.MLkNN", "mulan.classifier.meta.HOMER",
				"mulan.classifier.meta.RAkEL", "mulan.classifier.transformation.EnsembleOfPrunedSets",
				"mulan.classifier.transformation.EnsembleOfClassifierChains",
				"mulan.classifier.transformation.MultiLabelStacking", "mulan.classifier.transformation.PrunedSets",
				"mulan.classifier.transformation.ClassifierChain" };

		String nombreTransformacion = "";
		String nombreClasificador = "";

		MultiLabelLearner baseClassifier = null;
		MIMLtoML transformMethod = null;

		for (String metodoTransformacion : metodosTransformacion) {

			nombreTransformacion = lastToken(metodoTransformacion, "\\.");

			for (String clasificadorBase : clasificadoresBase) {

				nombreClasificador = lastToken(clasificadorBase, "\\.");

				try {

					// Clasificador ML

					Class<? extends MultiLabelLearner> classifierClass = (Class<? extends MultiLabelLearner>) Class
							.forName(clasificadorBase);
					baseClassifier = (MultiLabelLearner) classifierClass.newInstance();

					// Metodo de transformación
					Class<? extends MIMLtoML> transformerClass = (Class<? extends MIMLtoML>) Class
							.forName(metodoTransformacion);
					transformMethod = transformerClass.newInstance();

					MIMLClassifierML classifier = new MIMLClassifierML(baseClassifier, transformMethod);
					IEvaluator evaluator = loader.loadEvaluator();
					evaluator.runExperiment(classifier);

					logger.info("FINALIZADO CORRECTAMENTE => \t" + nombreTransformacion + "  -  " + nombreClasificador
							+ "\n");

				} catch (Exception e) {
					logger.info("ERROR EN PRUEBA => \t" + nombreTransformacion + "  -  " + nombreClasificador + " - "
							+ e.getMessage() + "\n");
				}
			}
		}

	}

}
