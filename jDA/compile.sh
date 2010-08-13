#!/bin/bash
echo "Compiling files..."
javac -cp $DXC_HOME/Src/APIs/java/src DiagnosticAlgorithm.java Sensor.java ErrorFinder.java ComponentError.java
echo "done."

echo -e "\nPacking files..."
jar cvf DiagnosticAlgorithm.jar *.class
