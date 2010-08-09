import java.io.*;

public class MyEvaluator {
				public static void main(String[] args) {
								System.out.println();
								Runtime r = Runtime.getRuntime();
								try {
												Process p = r.exec("Evaluator");
												InputStream in = p.getInputStream();
												BufferedInputStream buf = new BufferedInputStream(in);
												InputStreamReader inread = new InputStreamReader(buf);
												BufferedReader bufferedreader = new BufferedReader(inread);

												// Read the ls output
												String line;
												while ((line = bufferedreader.readLine()) != null) {
																if(line.indexOf("Recovery Cost")!=-1) {
																				System.out.println(line + "\n");
																				return;
																}
												}
												// Check for ls failure
												try {
																if (p.waitFor() != 0) {
																				System.err.println("exit value = " + p.exitValue());
																}
												} catch (InterruptedException e) {
																System.err.println(e);
												} finally {
																// Close the InputStream
																bufferedreader.close();
																inread.close();
																buf.close();
																in.close();
												}
								} catch (IOException e) {
												System.err.println(e.getMessage());
								}
				}
}
