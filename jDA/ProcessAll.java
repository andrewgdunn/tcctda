import java.io.*;

public class ProcessAll {
				public static void main(String[] args) {
					Runtime r = Runtime.getRuntime();
					InputStream in = null;
					try {
					    in = new FileInputStream("tempData");
					    BufferedReader reader = new BufferedReader(new InputStreamReader(in));
					    String line = null;
					    while ((line = reader.readLine()) != null) {
					        String[] parts = line.split(" ");
					    	
					        Process p = r.exec("./load2.sh " + parts[0]);
					        Thread.sleep(500);
					        
					        int time = processFile();
					        
					        double time2 = Double.parseDouble(parts[1]);
					        if(time2 != -1)
					        	time2 *= 1000;
					        
					        System.out.println( time + " - " + time2 + " -- " + Math.abs(time-time2));
					    }
					    if (in != null) in.close();
					} catch (Exception x) {
					    System.err.println(x);
					} 

					
					//System.out.println(processFile("ScenarioLoader -a"));
				}
				
				private static int processFile() {
								Runtime r = Runtime.getRuntime();
								try {
												Process p = r.exec("ScenarioLoader -a");
												InputStream in = p.getInputStream();
												BufferedInputStream buf = new BufferedInputStream(in);
												InputStreamReader inread = new InputStreamReader(buf);
												BufferedReader bufferedreader = new BufferedReader(inread);

												// Read the ls output
												String line;
												while ((line = bufferedreader.readLine()) != null) {
																if(line.indexOf("Error at")!=-1) {
																	int si = line.indexOf("Error at") + "Error at ".length();
																	int ei = si + line.substring(si).indexOf(" ");
																	return Integer.parseInt(line.substring(si,ei));
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
								return -1;
				}
}
