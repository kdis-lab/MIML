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

package es.uco.kdis.miml.report;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import es.uco.kdis.miml.core.IConfiguration;
import mulan.evaluation.measure.Measure;

/**
 * Abstract class for a MIMLReport.
 * 
 * @author Alvaro A. Belmonte
 * @author Amelia Zafra
 * @author Eva Gibaja
 * @version 20180630
 */
public abstract class MIMLReport implements IReport, IConfiguration {

	/** The measures shown in the report. */
	protected List<String> measures = null;

	/** The name of the file where report is saved. */
	protected String filename;

	/**
	 * Gets the measures shown in the report.
	 *
	 * @return the measures
	 */
	public List<String> getMeasures() {
		return measures;
	}

	/**
	 * Sets the measures shown in the report.
	 *
	 * @param measures the new measures
	 * @throws Exception the exception
	 */
	public void setMeasures(List<String> measures) throws Exception {
		this.measures = measures;
	}

	/**
	 * Gets the filename.
	 *
	 * @return the filename
	 */
	public String getFilename() {
		return filename;
	}

	/**
	 * Sets the name of the file.
	 *
	 * @param filename the new filename
	 */
	public void setFilename(String filename) {
		this.filename = filename;
	}

	/**
	 * Filter measures.
	 *
	 * @param allMeasures all the measures which the evaluation has
	 * @return the list with the measures filtered
	 * @throws Exception the exception
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
	 */
	@Override
	public void saveReport(String report) throws FileNotFoundException {

		try (PrintWriter out = new PrintWriter(filename)) {

			out.println(report);
			out.close();
			System.out.println("Results saved in " + filename);
		}
	}

}
