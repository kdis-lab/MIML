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

import java.io.FileNotFoundException;

import miml.evaluation.IEvaluator;

// TODO: Auto-generated Javadoc
/**
 * Interface for generate reports with the format specified.
 * 
 * @author Álvaro A. Belmonte
 * @author Amelia Zafra
 * @author Eva Gibaja
 * @version 20180630
 */
public interface IReport {

	/**
	 * Convert to CSV the evaluator results.
	 *
	 * @param evaluator the evaluator with the measures
	 * 
	 * @return the string with CSV content
	 * 
	 * @throws Exception the exception
	 */
	@SuppressWarnings("rawtypes")
	public String toCSV(IEvaluator evaluator) throws Exception;

	/**
	 * Convert to plain text the evaluator results.
	 *
	 * @param evaluator the evaluator with the measures
	 * 
	 * @return the string with the content
	 * 
	 * @throws Exception the exception
	 */
	@SuppressWarnings("rawtypes")
	public String toString(IEvaluator evaluator) throws Exception;

	/**
	 * Save in a file the specified report.
	 *
	 * @param report the formatted string to be saved
	 * 
	 * @throws FileNotFoundException the file not found exception
	 */
	public void saveReport(String report) throws FileNotFoundException;

}
