package gov.dod.army.rdecom.tardec.tcctda;

import org.dxc.api.datatypes.*;

import java.util.Vector;

/**
 * 
 * @author Jeremy Mange, Michael Duffy, Andrew Dunn 
 * @see License and contact information in project root
 * @version 0.0.1
 *
 * Used for storing sensor information and performing basic mathematics over selected internal data.
 * 
 * @todo Currently the data and timestamps are stored in vectors, realistically they should be in a multidimensional structure to avoid issues with concurrency.
 * @todo Need unit testing, the maths may be off by some amount of human error in here!
 */
public class Sensor {
	public String id;
	public Vector<Value> data;
	public Vector<Long> timestamps;
	public boolean hitsZero;
	
	private double noiseFloor = 0.3;
	private int largePositiveNumber = 99999999;
	private int largeNegativeNumber = -99999999;
	
	/**
	 * Instantiates a Sensor, requires only the identifier
	 * @param sensorID -- Identifier of Sensor
	 */
	public Sensor(String sensorID) {
		id = sensorID;
		data = new Vector<Value>();
		timestamps = new Vector<Long>();
		hitsZero = false;
	}
	
	/**
	 * Adds a data element to the data vector and a timestamp element to the timestamp vector
	 * @param value -- Data value
	 * @param timestamp -- Timestamp correlating to specific Data value
	 */
	public void addData(Value value, long timestamp) {
		// make all Integers into Reals
		if(value instanceof IntegerValue)
			value = Value.v(new Double(((IntegerValue)value).get()));
		data.add(value);
		timestamps.add(timestamp);
		
		// In some scenarios the noise will make it so that zero is never truely achieved, noiseFloor is our allowable range
		if( (value instanceof IntegerValue || value instanceof RealValue) && ( (RealValue) value).get() < noiseFloor ) {
			// Mark it zero donny.
			hitsZero = true;
		}
	}
	
	/**
	 * Calculates and returns the mean from the starting index through the number of samples provided
	 * @param start -- Starting index
	 * @param numSamples -- Number of samples to use when calculating the mean
	 * @return Mean throughout the Sensor Data from start through number of required samples
	 */
	public double meanThrough(int start, int numSamples) {
		double mean = 0;
		for(int index = start; index < numSamples; index++) {
			mean += ( (RealValue) data.elementAt(index)).get();
		}			
		mean /= (numSamples - start);
		return mean;
	}
	
	/**
	 * Calculates and returns the standard deviation from the starting index through the number of samples provided
	 * @param start -- Starting index
	 * @param numSamples -- Number of samples to use when calculating the mean
	 * @return Standard Deviation throughout the Sensor Data from start through the number of required samples
	 */
	public double stdThrough(int start, int numSamples) {
		double mean = meanThrough(0, numSamples);
		double std = 0;
		for(int index = start; index < numSamples; index++) {
			std += Math.pow(Math.abs(( (RealValue) data.elementAt(index)).get() - mean), 2);
		}			
		std = Math.sqrt(std/( (double) (numSamples-start)-1));
		return std;
	}

	/**
	 * Calculates the Minimum throughout the number of samples
	 * @param start -- Starting Index
	 * @param numSamples -- Number of samples to use when calculating minimum
	 * @return Minumum
	 */
	public double minThrough(int start, int numSamples) {
		double minimum = largePositiveNumber;
		double indexValue;
		for(int index = start; index < numSamples; index++) {
			indexValue = ((RealValue)data.elementAt(index)).get();
			if(indexValue < minimum)
				minimum = indexValue;
		}			
		return minimum;
	}
	
	/**
	 * Calculates the Maximum throughout the number of samples
	 * @param start -- Starting Index
	 * @param numSamples -- Number of samples to use when calculating maximum
	 * @return Maximum
	 */
	public double maxThrough(int start, int numSamples) {
		double maximum = largeNegativeNumber;
		double indexValue;
		for(int index = start; index < numSamples; index++) {
			indexValue = ((RealValue)data.elementAt(index)).get();
			if(indexValue > maximum)
				maximum = indexValue;
		}			
		return maximum;
	}
	
	/**
	 * Removes the outlier data instances, one for high and one for low through the data set
	 */
	public void removeOutliers() {
		// removes the maximum and minimum values from the data (necessary for one scenario)
		if(data.elementAt(0) instanceof RealValue) {
			double min = largePositiveNumber;
			double max= largeNegativeNumber;
			int minIndex=-1; 
			int maxIndex=-1;
			for(int index = 0; index < data.size(); index++) {
				double valueIndex = ( (RealValue) data.elementAt(index)).get();
				if(valueIndex >= max) {
					max = valueIndex;
					maxIndex = index;
				}
				if(valueIndex <= min) {
					min=valueIndex;
					minIndex=index;
				}
			}
			if(minIndex > -1)
				data.removeElementAt(minIndex);
				timestamps.removeElementAt(minIndex);
			// index for maxIndex will have decreased from the removal
			if(maxIndex > 0)
				data.removeElementAt(maxIndex-1);
				timestamps.removeElementAt(maxIndex-1);
		}
	}
}