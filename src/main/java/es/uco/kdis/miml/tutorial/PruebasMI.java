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

import es.uco.kdis.miml.classifiers.miml.mimlTOmi.MIMLClassifierMI;
import es.uco.kdis.miml.core.ConfigLoader;
import es.uco.kdis.miml.evaluation.IEvaluator;
import mulan.classifier.MultiLabelLearner;
import weka.classifiers.Classifier;
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
public class PruebasMI {

	public static String lastToken(String str, String separatorRegex) {
		String tokens[] = str.split(separatorRegex);
		return tokens[tokens.length - 1];
	}

	static Logger logger = Logger.getLogger("MyLog");
	static FileHandler fh;

	/**
	 * The main method to configure and run an algorithm.
	 *
	 * @param args
	 *            the arguments(route of config file with the option -c)
	 * @throws Exception
	 * @throws ConfigurationException
	 */
	@SuppressWarnings({ "unused", "unchecked" })
	public static void main(String[] args) throws ConfigurationException, Exception {

		try {

			// This block configure the logger with handler and formatter
			fh = new FileHandler("logs/MyLogFile.log");
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
				"es.uco.kdis.miml.mimlclassifier.mimlTOmi.transformation.MIMLLabelPowerset",
				"mulan.classifier.transformation.BinaryRelevance", "mulan.classifier.transformation.AdaBoostMH",
				"mulan.classifier.transformation.CalibratedLabelRanking",
				"mulan.classifier.transformation.ClassifierChain",
				"mulan.classifier.transformation.EnsembleOfClassifierChains",
				"mulan.classifier.transformation.EnsembleOfPrunedSets",
				"mulan.classifier.transformation.IncludeLabelsClassifier",
				"mulan.classifier.transformation.MultiClassLearner",
				"mulan.classifier.transformation.MultiLabelStacking", "mulan.classifier.transformation.Pairwise",
				"mulan.classifier.transformation.PPT", "mulan.classifier.transformation.PrunedSets",
				"mulan.classifier.transformation.TwoStageClassifierChainArchitecture",
				"mulan.classifier.transformation.TwoStagePrunedClassifierChainArchitecture",
				"mulan.classifier.transformation.TwoStageVotingArchitecture" };

		String[] clasificadoresBase = new String[] { "weka.classifiers.mi.CitationKNN", "weka.classifiers.mi.MDD",
				"weka.classifiers.mi.MIDD", "weka.classifiers.mi.MIBoost", "weka.classifiers.mi.MIEMDD",
				"weka.classifiers.mi.MILR", "weka.classifiers.mi.MIOptimalBall", "weka.classifiers.mi.MIRI",
				"weka.classifiers.mi.MISMO", "weka.classifiers.mi.MISVM", "weka.classifiers.mi.MITI",
				"weka.classifiers.mi.MIWrapper", "weka.classifiers.mi.SimpleMI" };

		String nombreTransformacion = "";
		String nombreClasificador = "";

		for (String metodoTransformacion : metodosTransformacion) {

			nombreTransformacion = lastToken(metodoTransformacion, "\\.");

			for (String clasificadorBase : clasificadoresBase) {

				nombreClasificador = lastToken(clasificadorBase, "\\.");

				try {

					// Se crea el clasificador MIMLtoMI asignando clasificador base y método de
					// transformación
					MultiLabelLearner transformationClassifier = null;

					// Get the transformation classifier method
					String transformName = metodoTransformacion;
					// Instantiate the transformation classifier class used in the experiment
					Class<? extends MultiLabelLearner> clsClass = (Class<? extends MultiLabelLearner>) Class
							.forName(transformName);

					// Get the name of the MI base classifier class
					String baseName = clasificadorBase;

					// Instance class
					Class<? extends Classifier> baseClassifier = (Class<? extends Classifier>) Class.forName(baseName);

					transformationClassifier = clsClass.getConstructor(Classifier.class)
							.newInstance(baseClassifier.newInstance());

					MIMLClassifierMI classifier = new MIMLClassifierMI(transformationClassifier);
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
