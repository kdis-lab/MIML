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

package miml.report;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.configuration2.Configuration;

import miml.core.ConfigParameters;
import miml.data.MIMLInstances;
import miml.evaluation.EvaluatorCV;
import miml.evaluation.EvaluatorHoldout;
import miml.evaluation.IEvaluator;
import mulan.evaluation.Evaluation;
import mulan.evaluation.MultipleEvaluation;
import mulan.evaluation.measure.MacroAverageMeasure;
import mulan.evaluation.measure.Measure;

/**
 * Class used to generate reports with the format specified.
 * 
 * @author Alvaro A. Belmonte
 * @author Amelia Zafra
 * @author Eva Gibaja
 * @version 20180630
 */
public class BaseMIMLReport extends MIMLReport {

	/**
	 * Basic constructor to initialize the report.
	 *
	 * @param measures The list of selected measures which is going to be shown in
	 *                 the report.
	 * @param filename The filename where the report's will be saved.
	 * @param std      Whether the standard deviation of measures will be shown or
	 *                 not (only valid for cross-validation evaluator).
	 * @param labels   Whether the measures for each label will be shown (only valid
	 *                 for Macro-Averaged measures).
	 * @param header   Whether the header will be shown.
	 */
	public BaseMIMLReport(List<String> measures, String filename, boolean std, boolean labels, boolean header) {
		super();
	}

	/**
	 * No-argument constructor for xml configuration.
	 */
	public BaseMIMLReport() {
	}

	/**
	 * Read the cross-validation results and transform to CSV format.
	 *
	 * @param evaluator The evaluator.
	 * @return String with CSV content.
	 * @throws Exception To be handled in an upper level.
	 */
	protected String crossValidationToCSV(EvaluatorCV evaluator) throws Exception {

		MultipleEvaluation evaluationCrossValidation = evaluator.getEvaluation();
		MIMLInstances data = evaluator.getData();

		ArrayList<Evaluation> evaluations = evaluationCrossValidation.getEvaluations();
		StringBuilder sb = new StringBuilder();
		String measureName;

		// All evaluator measures
		List<Measure> measures = evaluations.get(0).getMeasures();
		// Measures selected by user
		if (this.measures != null)
			measures = filterMeasures(measures);

		if (this.header) {
			if (ConfigParameters.getIsTransformation()) {
				// Write header
				sb.append("Algorithm," + "Classifier," + "Transform method," + "Dataset," + "ConfigurationFile,"
						+ "Train_time_ms(avg),");
			} else {
				// Write header
				sb.append("Algorithm," + "Dataset," + "ConfigurationFile," + "Train_time_ms(avg),");
			}

			if (this.std) {
				sb.append("Train_time_ms(std),");
			}

			sb.append("Test_time_ms(avg),");

			if (this.std) {
				sb.append("Test_time_ms(std),");
			}

			// Write measure's names
			for (Measure m : measures) {
				measureName = m.getName();
				sb.append(measureName + ",");
				// If std label is activated standard deviation is written
				if (this.std) {

					sb.append(measureName + " Std,");

					if (m instanceof MacroAverageMeasure && this.labels) {

						for (int i = 0; i < data.getNumLabels(); i++) {
							sb.append(measureName + "-" + data.getDataSet().attribute(data.getLabelIndices()[i]).name()
									+ ",");
							sb.append(measureName + "-" + data.getDataSet().attribute(data.getLabelIndices()[i]).name()
									+ " Std,");
						}
					} else if (m instanceof MacroAverageMeasure) {
						for (int i = 0; i < data.getNumLabels(); i++) {
							sb.append(measureName + "-" + data.getDataSet().attribute(data.getLabelIndices()[i]).name()
									+ ",");
						}
					}
				}

				else if (m instanceof MacroAverageMeasure && this.labels) {
					for (int i = 0; i < data.getNumLabels(); i++) {
						sb.append(measureName + "-" + data.getDataSet().attribute(data.getLabelIndices()[i]).name()
								+ ",");
					}
				}
			}
			sb.setLength(sb.length() - 1);
			sb.append(System.getProperty("line.separator"));
		}

		if (ConfigParameters.getIsTransformation()) {
			// Write header
			sb.append(ConfigParameters.getAlgorithmName() + "," + ConfigParameters.getClassifierName() + ","
					+ ConfigParameters.getTransformationMethod() + "," + ConfigParameters.getDataFileName() + ","
					+ ConfigParameters.getConfigFileName() + "," + evaluator.getAvgTrainTime() + ",");
		} else {
			// Write header
			sb.append(ConfigParameters.getAlgorithmName() + "," + ConfigParameters.getDataFileName() + ","
					+ ConfigParameters.getConfigFileName() + "," + evaluator.getAvgTrainTime() + ",");
		}

		if (this.std) {
			sb.append(evaluator.getStdTrainTime() + ",");
		}

		sb.append(evaluator.getAvgTestTime() + ",");

		if (this.std) {
			sb.append(evaluator.getStdTestTime() + ",");
		}

		// Write mean and std(optional) for each measure
		for (Measure m : measures) {
			measureName = m.getName();
			sb.append(evaluationCrossValidation.getMean(measureName) + ",");

			if (this.std) {

				sb.append(evaluationCrossValidation.getStd(measureName) + ",");

				if (m instanceof MacroAverageMeasure && this.labels) {

					for (int i = 0; i < data.getNumLabels(); i++) {
						sb.append(evaluationCrossValidation.getMean(measureName, i) + ",");
						sb.append(evaluationCrossValidation.getStd(measureName, i) + ",");
					}
				} else if (m instanceof MacroAverageMeasure) {
					for (int i = 0; i < data.getNumLabels(); i++) {
						sb.append(evaluationCrossValidation.getMean(measureName, i) + ",");
					}
				}
			}

			else if (m instanceof MacroAverageMeasure && this.labels) {
				for (int i = 0; i < data.getNumLabels(); i++) {
					sb.append(evaluationCrossValidation.getMean(measureName, i) + ",");
				}
			}

		}
		sb.setLength(sb.length() - 1);
		sb.append(System.getProperty("line.separator"));
		return sb.toString();
	}

	/**
	 * Read the holdout results and transform to CSV format.
	 *
	 * @param evaluator The evaluator.
	 * @return String with CSV content.
	 * @throws Exception To be handled in an upper level
	 */
	protected String holdoutToCSV(EvaluatorHoldout evaluator) throws Exception {

		Evaluation evaluationHoldout = evaluator.getEvaluation();
		MIMLInstances data = evaluator.getData();

		StringBuilder sb = new StringBuilder();
		String measureName;

		if(this.std) {
			System.out.println("[WARNING]: standardDeviation is setted true, but in holdout evaluation is not possible calculate std value");
		}
		
		// All evaluator measures
		List<Measure> measures = evaluationHoldout.getMeasures();
		// Measures selected by user
		if (this.measures != null)
			measures = filterMeasures(measures);

		if (this.header) {
			if (ConfigParameters.getIsTransformation()) {
				// Write header
				sb.append("Algorithm," + "Classifier," + "Transform method," + "Dataset," + "ConfigurationFile,"
						+ "Train_time_ms," + "Test_time_ms,");
			} else {
				// Write header
				sb.append("Algorithm," + "Dataset," + "ConfigurationFile," + "Train_time_ms," + "Test_time_ms,");
			}
			// Write measure's names
			for (Measure m : measures) {
				measureName = m.getName();
				sb.append(measureName + ",");

				if (m instanceof MacroAverageMeasure && this.labels) {

					for (int i = 0; i < data.getNumLabels(); i++) {
						sb.append(measureName + "-" + data.getDataSet().attribute(data.getLabelIndices()[i]).name()
								+ ",");
					}
				}
			}
			sb.setLength(sb.length() - 1);
			sb.append(System.getProperty("line.separator"));
		}

		if (ConfigParameters.getIsTransformation()) {
			// Write header
			sb.append(ConfigParameters.getAlgorithmName() + "," + ConfigParameters.getClassifierName() + ","
					+ ConfigParameters.getTransformationMethod() + "," + ConfigParameters.getDataFileName() + ","
					+ ConfigParameters.getConfigFileName() + "," + evaluator.getTrainTime() + ","
					+ evaluator.getTestTime() + ",");
		} else {
			// Write header
			sb.append(ConfigParameters.getAlgorithmName() + "," + ConfigParameters.getDataFileName() + ","
					+ ConfigParameters.getConfigFileName() + "," + evaluator.getTrainTime() + ","
					+ evaluator.getTestTime() + ",");
		}

		// Write value for each measure
		for (Measure m : measures) {
			sb.append(m.getValue() + ",");

			if (m instanceof MacroAverageMeasure && this.labels) {

				for (int i = 0; i < data.getNumLabels(); i++) {
					sb.append(((MacroAverageMeasure) m).getValue(i) + ",");
				}
			}

		}

		sb.setLength(sb.length() - 1);
		sb.append(System.getProperty("line.separator"));
		return sb.toString();
	}

	/**
	 * Read the cross-validation results and transform to plain text.
	 *
	 * @param evaluator The evaluator.
	 * @return String with the content.
	 * @throws Exception To be handled in an upper level
	 */
	protected String crossValidationToString(EvaluatorCV evaluator) throws Exception {

		MultipleEvaluation evaluationCrossValidation = evaluator.getEvaluation();
		MIMLInstances data = evaluator.getData();

		ArrayList<Evaluation> evaluations = evaluationCrossValidation.getEvaluations();
		StringBuilder sb = new StringBuilder();
		String measureName;

		// All evaluator measures
		List<Measure> measures = evaluations.get(0).getMeasures();
		// Measures selected by user
		if (this.measures != null)
			measures = filterMeasures(measures);

		if (this.header) {
			sb.append("Algorithm: " + ConfigParameters.getAlgorithmName() + System.getProperty("line.separator"));
			sb.append("Classifier: " + ConfigParameters.getClassifierName() + System.getProperty("line.separator"));
			sb.append("Transform method: " + ConfigParameters.getTransformationMethod()
					+ System.getProperty("line.separator"));
			sb.append("Dataset: " + ConfigParameters.getDataFileName() + System.getProperty("line.separator"));
			sb.append("Config File: " + ConfigParameters.getConfigFileName() + System.getProperty("line.separator"));
		}

		sb.append("Train time avg (ms): " + evaluator.getAvgTrainTime() + System.getProperty("line.separator"));

		if (this.std) {
			sb.append("Train time std (ms): " + evaluator.getStdTrainTime() + System.getProperty("line.separator"));
		}

		sb.append("Test time avg (ms): " + evaluator.getAvgTestTime() + System.getProperty("line.separator"));

		if (this.std) {
			sb.append("Test time std (ms): " + evaluator.getStdTestTime() + System.getProperty("line.separator"));
		}

		for (Measure m : measures) {
			measureName = m.getName();
			sb.append(measureName);
			sb.append(": ");

			sb.append(String.format("%.4f", evaluationCrossValidation.getMean(measureName)));

			if (this.std) {
				sb.append("\u00B1");
				sb.append(String.format("%.4f", evaluationCrossValidation.getStd(measureName)));
				sb.append(System.getProperty("line.separator"));
				if (m instanceof MacroAverageMeasure && this.labels) {

					for (int i = 0; i < data.getNumLabels(); i++) {
						sb.append(measureName + " - " + data.getDataSet().attribute(data.getLabelIndices()[i]).name()
								+ ": ");
						sb.append(String.format("%.4f", evaluationCrossValidation.getMean(measureName, i)));
						sb.append("\u00B1");
						sb.append(String.format("%.4f", evaluationCrossValidation.getStd(measureName, i)));
						sb.append(" ");
						sb.append(System.getProperty("line.separator"));
					}
				}
			} else if (m instanceof MacroAverageMeasure && this.labels) {
				for (int i = 0; i < data.getNumLabels(); i++) {
					sb.append(
							measureName + " - " + data.getDataSet().attribute(data.getLabelIndices()[i]).name() + ": ");
					sb.append(String.format("%.4f", evaluationCrossValidation.getMean(measureName, i)));
					sb.append(System.getProperty("line.separator"));
				}
			} else {
				sb.append(System.getProperty("line.separator"));
			}
		}
		return sb.toString();
	}

	/**
	 * Read the holdout results and transform to plain text.
	 *
	 * @param evaluator The evaluator.
	 * @return String with the content.
	 * @throws Exception To be handled in an upper level.
	 */
	protected String holdoutToString(EvaluatorHoldout evaluator) throws Exception {

		if(this.std) {
			System.out.println("[WARNING]: standardDeviation is setted true, but in holdout evaluation is not possible  to calculate std value");
		}
		
		Evaluation evaluationHoldout = evaluator.getEvaluation();
		MIMLInstances data = evaluator.getData();
		StringBuilder sb = new StringBuilder();

		// All evaluator measures
		List<Measure> measures = evaluationHoldout.getMeasures();
		// Measures selected by user
		if (this.measures != null)
			measures = filterMeasures(measures);

		if (this.header) {
			sb.append("Algorithm: " + ConfigParameters.getAlgorithmName() + System.getProperty("line.separator"));
			sb.append("Classifier: " + ConfigParameters.getClassifierName() + System.getProperty("line.separator"));
			sb.append("Transform method: " + ConfigParameters.getTransformationMethod()
					+ System.getProperty("line.separator"));
			sb.append("Dataset: " + ConfigParameters.getDataFileName() + System.getProperty("line.separator"));
			sb.append("Config File: " + ConfigParameters.getConfigFileName() + System.getProperty("line.separator"));
		}

		sb.append("Train time (ms): " + evaluator.getTrainTime() + System.getProperty("line.separator"));
		sb.append("Test time (ms): " + evaluator.getTestTime() + System.getProperty("line.separator"));

		for (Measure m : measures) {
			sb.append(m);
			if (m instanceof MacroAverageMeasure && this.labels) {
				sb.append(System.getProperty("line.separator"));
				for (int i = 0; i < data.getNumLabels(); i++) {
					sb.append(data.getDataSet().attribute(data.getLabelIndices()[i]).name());
					sb.append(": ");
					sb.append(String.format("%.4f", ((MacroAverageMeasure) m).getValue(i)));
					sb.append(" ");
				}
			}
			sb.append(System.getProperty("line.separator"));
		}
		return sb.toString();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see report.IReport#toCSV(evaluation.IEvaluator)
	 */
	@SuppressWarnings("rawtypes")
	@Override
	public String toCSV(IEvaluator evaluator) throws Exception {

		if (evaluator instanceof EvaluatorCV) {
			return crossValidationToCSV((EvaluatorCV) evaluator);
		} else {
			return holdoutToCSV((EvaluatorHoldout) evaluator);
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see report.IReport#toString(evaluation.IEvaluator)
	 */
	@SuppressWarnings("rawtypes")
	@Override
	public String toString(IEvaluator evaluator) throws Exception {
		if (evaluator instanceof EvaluatorCV) {
			return crossValidationToString((EvaluatorCV) evaluator);
		} else {
			return holdoutToString((EvaluatorHoldout) evaluator);
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * core.IConfiguration#configure(org.apache.commons.configuration.Configuration)
	 */
	@Override
	public void configure(Configuration configuration) {
		
		this.filename = configuration.getString("fileName");
		this.std = configuration.getBoolean("standardDeviation", false);
		this.header = configuration.getBoolean("header", true);
		
		this.labels = configuration.getBoolean("measures[@perLabel]", true);

		int measuresLength = configuration.getList("measures.measure").size();

		if (measuresLength > 0) {
			measures = new ArrayList<String>();

			for (int i = 0; i < measuresLength; ++i) {
				measures.add(configuration.getString("measures.measure(" + i + ")"));
			}
		}

	}

}
