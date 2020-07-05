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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import miml.core.IConfiguration;
import mulan.evaluation.measure.Measure;

/**
 * Abstract class for a MIMLReport.
 * 
 * @author Alvaro A. Belmonte
 * @author Amelia Zafra
 * @author Eva Gibaja
 * @version 20190502
 */
public abstract class MIMLReport implements IReport, IConfiguration {

	/** The measures shown in the report. */
	protected List<String> measures = null;

	/** The name of the file where report is saved. */
	protected String filename;

	/** If measures' standard deviation are shown. */
	protected boolean std;

	/** If macro measures are broken down by labels. */
	protected boolean labels;
	
	/** If the header is going to be printed. */
	protected boolean header;

	/**
	 * Basic constructor to initialize the report.
	 *
	 * @param measures 		The list of selected measures which is going to be shown in the report.
	 * @param filename     	The filename where the report's will be saved.
	 * @param std         	Whether the standard deviation of measures will be shown or not (only valid for cross-validation evaluator).
	 * @param labels 		Whether the measures for each label will be shown (only valid for Macro-Averaged measures).
	 * @param header 		Whether the header will be shown.
	 */
	public MIMLReport(List<String> measures, String filename, boolean std, boolean labels, boolean header) {
		this.measures = measures;
		this.filename = filename;
		this.std = std;
		this.labels = labels;
		this.header = header;
	}
	
	/**
	 * No-argument constructor for xml configuration.
	 */
	public MIMLReport() {
		
	}
	
	/**
	 * Filter measures chosen to be shown in the experiment report.
	 *
	 * @param allMeasures All the measures which the evaluation has
	 * @return List with the measures filtered
	 * @throws Exception To be handled in an upper level.
	 */
	protected List<Measure> filterMeasures(List<Measure> allMeasures) throws Exception {

		List<Measure> measures = new ArrayList<Measure>();

		for (String s : this.measures) {

			for (Measure m : allMeasures) {

				if (m.getName().equals(s))
					measures.add(m.makeCopy());
			}
		}

		return measures;
	}

	/**
	 * Save in a file the specified report.
	 *
	 * @param report The report.
	 * @throws FileNotFoundException To be handled in an upper level.
	 */
	@Override
	public void saveReport(String report) throws FileNotFoundException {

		File file = new File(filename);
		file.getParentFile().mkdirs();

		try {
			file.createNewFile();
			Files.write(Paths.get(filename), report.getBytes(), StandardOpenOption.APPEND);
			System.out.println("" + new Date() + ": " + "Experiment results saved in " + filename);

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Gets the measures shown in the report.
	 *
	 * @return The measures.
	 */
	public List<String> getMeasures() {
		return measures;
	}

	/**
	 * Sets the measures shown in the report.
	 *
	 * @param measures The new measures.
	 * @throws Exception To be handled in an upper level.
	 */
	public void setMeasures(List<String> measures) throws Exception {
		this.measures = measures;
	}

	/**
	 * Gets the filename.
	 *
	 * @return The filename.
	 */
	public String getFilename() {
		return filename;
	}

	/**
	 * Sets the name of the file.
	 *
	 * @param filename The new filename
	 */
	public void setFilename(String filename) {
		this.filename = filename;
	}

	/**
	 * Checks if std is going to be shown (only cross-validation).
	 *
	 * @return True, if std is going to be shown.
	 */
	public boolean isStd() {
		return std;
	}

	/**
	 * Sets if the std is going to be shown (only cross-validation).
	 *
	 * @param std The new std configuration.
	 */
	public void setStd(boolean std) {
		this.std = std;
	}

	/**
	 * Checks if measure for each label (macro-averaged measures) is shown.
	 *
	 * @return True, if measure for each label is shown.
	 */
	public boolean isLabels() {
		return labels;
	}

	/**
	 * Sets if measure for each label (macro-averaged measures) is shown.
	 * 
	 * @param labels The new labels configuration.
	 */
	public void setLabels(boolean labels) {
		this.labels = labels;
	}

	/**
	 * Checks if header is shown.
	 *
	 * @return True, if header is shown.
	 */
	public boolean isHeader() {
		return header;
	}
	
	/**
	 * Sets if header is shown.
	 * 
	 * @param header The new header configuration.
	 */
	public void setHeader(boolean header) {
		this.header = header;
	}

}
